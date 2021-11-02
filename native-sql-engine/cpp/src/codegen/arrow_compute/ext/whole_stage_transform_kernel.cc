/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <arrow/array.h>
#include <arrow/compute/api.h>
#include <arrow/pretty_print.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <arrow/util/bit_util.h>
#include <gandiva/node.h>
#include <gandiva/projector.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "codegen/arrow_compute/ext/codegen_common.h"
#include "codegen/arrow_compute/ext/kernels_ext.h"
#include "codegen/common/hash_relation.h"
#include "utils/macros.h"

namespace sparkcolumnarplugin {
namespace codegen {
namespace arrowcompute {
namespace extra {

using ArrayList = std::vector<std::shared_ptr<arrow::Array>>;

///////////////  WholeStageTransform  ////////////////
class WholeStageTransformKernel::Impl {
 public:
  Impl(arrow::compute::ExecContext* ctx,
       const std::vector<std::shared_ptr<arrow::Field>>& input_field_list,
       const std::shared_ptr<gandiva::Node>& root_node,
       const std::vector<std::shared_ptr<arrow::Field>>& output_field_list)
      : ctx_(ctx), root_node_(root_node) {
    int hash_relation_idx = 0;
    enable_time_metrics_ = GetEnableTimeMetrics();
  }

  arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
    *out = std::make_shared<WholeStageTransformResultIterator>(ctx_, root_node_);
    return arrow::Status::OK();
  }

  std::string GetSignature() { return signature_; }

 private:
  arrow::compute::ExecContext* ctx_;
  arrow::MemoryPool* pool_;
  std::shared_ptr<gandiva::Node> root_node_;
  std::string signature_;
  bool enable_time_metrics_;
  std::vector<std::shared_ptr<GandivaProjector>> gandiva_projector_list_;

  class WholeStageTransformResultIterator : public ResultIterator<arrow::RecordBatch> {
   public:
    WholeStageTransformResultIterator(arrow::compute::ExecContext* ctx,
                                      const std::shared_ptr<gandiva::Node>& root_node)
        : ctx_(ctx), root_node_(root_node) {}

    arrow::Status ApplyWSCGRules(
        const std::vector<std::shared_ptr<KernelInfo>>& kernel_info_list) {
      auto agg_kernel_name = "HashAggregateKernel";
      auto cond_project_kernel_name = "CondProjectKernel";

      int kernel_idx = kernel_info_list.size() - 1;
      while (kernel_idx > 0) {
        auto kernel_name = kernel_info_list[kernel_idx]->kernel_name;
        if (kernel_name == agg_kernel_name) {
          if (kernel_idx >= 1 &&
              kernel_info_list[kernel_idx - 1]->kernel_name == cond_project_kernel_name) {
            kernel_info_list[kernel_idx]->setDoCodegen(true);
            kernel_info_list[kernel_idx - 1]->setDoCodegen(true);
          }
        }
        kernel_idx -= 1;
      }
      return arrow::Status::OK();
    }

    arrow::Status CreateWSCGIters(
        std::vector<std::shared_ptr<KernelInfo>>* kernel_info_list) {
      std::vector<std::shared_ptr<KernelInfo>> new_kernel_info_list;
      int kernel_idx = 0;
      int32_t kernel_size = (*kernel_info_list).size();

      while (kernel_idx < kernel_size) {
        std::vector<std::shared_ptr<KernelInfo>> kernel_info_in_wscg;
        while (kernel_idx < kernel_size && (*kernel_info_list)[kernel_idx]->do_wscg) {
          kernel_info_in_wscg.push_back((*kernel_info_list)[kernel_idx]);
          kernel_idx += 1;
        }
        if (kernel_info_in_wscg.size() == 0) {
          new_kernel_info_list.push_back((*kernel_info_list)[kernel_idx]);
          kernel_idx += 1;
        } else {
          auto input_field_list = kernel_info_in_wscg[0]->input_fields;
          auto output_field_list =
              kernel_info_in_wscg[kernel_info_in_wscg.size() - 1]->res_fields;
          std::vector<std::shared_ptr<KernalBase>> kernels_in_wscg;
          std::vector<std::shared_ptr<ResultIteratorBase>> dep_iter_list;
          for (auto kernel_info : kernel_info_in_wscg) {
            kernels_in_wscg.push_back(kernel_info->kernel);
            for (auto dep : kernel_info->dependent_iter_list) {
              dep_iter_list.push_back(dep);
            }
          }
          THROW_NOT_OK(LoadJITFunction(input_field_list, output_field_list,
                                       kernels_in_wscg, &wscg_handler_));
          auto res_schema = arrow::schema(output_field_list);
          std::shared_ptr<ResultIterator<arrow::RecordBatch>> wscg_iter;
          wscg_handler_->MakeResultIterator(res_schema, gandiva_projector_list_,
                                            &wscg_iter);
          std::shared_ptr<KernalBase> wscg_kernel =
              nullptr;  // fake kernel, will not be used
          auto kernel_info =
              std::make_shared<KernelInfo>(wscg_kernel, wscg_iter, input_field_list,
                                           output_field_list, "WSCG", dep_iter_list);
          new_kernel_info_list.push_back(kernel_info);
        }
      }
      (*kernel_info_list) = std::move(new_kernel_info_list);
      return arrow::Status::OK();
    }

    arrow::Status SetDependencies(const std::vector<std::shared_ptr<ResultIteratorBase>>&
                                      dependent_iter_list) override {
      // Parse the tree
      int hash_relation_idx = 0;
      int dep_iter_idx = dependent_iter_list.size() - 1;
      THROW_NOT_OK(ParseNodeTree(root_node_, dependent_iter_list, &hash_relation_idx,
                                 &dep_iter_idx, &kernel_info_list_));
      // Decide whether a kernel will be merged into wscg according to certain rules.
      // THROW_NOT_OK(ApplyWSCGRules(kernel_info_list_));

      // Create WSCG iters for the kernels merged into WSCG
      // THROW_NOT_OK(CreateWSCGIters(&kernel_info_list_));

      // Create kernels for filter and project according to whether to do codegen.
      THROW_NOT_OK(ParseCondProjectByCodegen(&kernel_info_list_));

      // Set dependencies
      for (auto kernel_info : kernel_info_list_) {
        auto deps = kernel_info->dependent_iter_list;
        if (deps.size() > 0) {
          kernel_info->res_iter->SetDependencies(deps);
        }
      }
      // Set child iter
      for (int i = kernel_info_list_.size() - 1; i > 0; i--) {
        kernel_info_list_[i]->res_iter->SetChildResIter(
            kernel_info_list_[i - 1]->res_iter);
      }
      // Set the fist input
      if (dep_iter_idx == 0) {
        auto typed_dependent =
            std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
                dependent_iter_list[0]);
        if (typed_dependent == nullptr) {
          throw std::runtime_error("casting on recordbatch iterator failed");
        }
        kernel_info_list_[0]->res_iter->SetChildResIter(typed_dependent);
      }
      last_iter_ = kernel_info_list_[kernel_info_list_.size() - 1]->res_iter;
      return arrow::Status::OK();
    }

    bool HasNext() override { return last_iter_->HasNext(); }

    arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out) override {
      return last_iter_->Next(out);
    }

   private:
    arrow::compute::ExecContext* ctx_;
    std::shared_ptr<gandiva::Node> root_node_;
    std::shared_ptr<ResultIterator<arrow::RecordBatch>> last_iter_;
    // WSCG Related variables
    bool is_smj_ = false;
    bool is_aggr_ = false;
    std::string aggr_finish_condition_;
    std::shared_ptr<CodeGenBase> wscg_handler_;
    std::string signature_;
    bool enable_time_metrics_;
    std::vector<std::shared_ptr<GandivaProjector>> gandiva_projector_list_;
    std::vector<std::string> aggr_out_length_idxs;
    std::vector<std::shared_ptr<KernelInfo>> kernel_info_list_;

    arrow::Status GetArguments(std::shared_ptr<gandiva::Node> node, int i,
                               gandiva::NodeVector* node_list) {
      auto function_node = std::dynamic_pointer_cast<gandiva::FunctionNode>(node);
      auto arg_node =
          std::dynamic_pointer_cast<gandiva::FunctionNode>(function_node->children()[i]);
      *node_list = arg_node->children();
      return arrow::Status::OK();
    }

    arrow::Status CreateKernelAndSetDep(
        std::shared_ptr<gandiva::Node> node,
        const std::vector<std::shared_ptr<ResultIteratorBase>>& dependent_iter_list,
        int* hash_relation_idx, int* dep_iter_idx, std::string* kernel_name_out,
        std::shared_ptr<KernalBase>* kernel_out,
        std::shared_ptr<ResultIterator<arrow::RecordBatch>>* iter_out,
        std::vector<std::shared_ptr<arrow::Field>>* input_fields_out,
        std::vector<std::shared_ptr<arrow::Field>>* res_fields_out,
        std::vector<std::shared_ptr<ResultIteratorBase>>* dependent_iter_list_out,
        std::shared_ptr<gandiva::FunctionNode>* function_node_out) {
      auto function_node = std::dynamic_pointer_cast<gandiva::FunctionNode>(node);
      auto func_name = function_node->descriptor()->name();
      if (func_name.compare(0, 22, "conditionedProbeArrays") == 0) {
        int join_type = 0;
        bool is_null_aware_anti_join = false;
        gandiva::NodeVector left_schema_list;
        RETURN_NOT_OK(GetArguments(function_node, 0, &left_schema_list));
        gandiva::NodeVector right_schema_list;
        RETURN_NOT_OK(GetArguments(function_node, 1, &right_schema_list));
        gandiva::NodeVector left_key_list;
        RETURN_NOT_OK(GetArguments(function_node, 2, &left_key_list));
        gandiva::NodeVector right_key_list;
        RETURN_NOT_OK(GetArguments(function_node, 3, &right_key_list));
        gandiva::NodeVector result_list;
        RETURN_NOT_OK(GetArguments(function_node, 4, &result_list));
        gandiva::NodeVector configuration_list;
        RETURN_NOT_OK(GetArguments(function_node, 5, &configuration_list));
        gandiva::NodePtr condition;
        if (function_node->children().size() > 6) {
          condition = function_node->children()[6];
        }
        if (func_name.compare("conditionedProbeArraysInner") == 0) {
          join_type = 0;
        } else if (func_name.compare("conditionedProbeArraysOuter") == 0) {
          join_type = 1;
        } else if (func_name.compare(0, 26, "conditionedProbeArraysAnti") == 0) {
          if (func_name.length() > 26 &&
              func_name.compare(0, 27, "conditionedProbeArraysAnti_") == 0) {
            auto lit = func_name.substr(27);
            is_null_aware_anti_join = (lit == "true" ? true : false);
          } else {
            is_null_aware_anti_join = false;
          }
          join_type = 2;
        } else if (func_name.compare("conditionedProbeArraysSemi") == 0) {
          join_type = 3;
        } else if (func_name.compare("conditionedProbeArraysExistence") == 0) {
          join_type = 4;
        }
        int cur_hash_relation_idx = *hash_relation_idx;
        *hash_relation_idx += 1;
        RETURN_NOT_OK(ConditionedProbeKernel::Make(
            ctx_, left_key_list, right_key_list, left_schema_list, right_schema_list,
            condition, join_type, is_null_aware_anti_join, result_list,
            configuration_list, cur_hash_relation_idx, kernel_out));
        std::vector<std::shared_ptr<arrow::Field>> input_fields;
        for (auto node : left_schema_list) {
          input_fields.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        std::vector<std::shared_ptr<arrow::Field>> res_fields;
        for (auto node : result_list) {
          res_fields.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        // get res iter and set dependency
        (*kernel_out)->MakeResultIterator(arrow::schema(res_fields), iter_out);
        std::vector<std::shared_ptr<ResultIteratorBase>> dependencies;
        dependencies.push_back(dependent_iter_list[*dep_iter_idx]);
        // (*iter_out)->SetDependencies(dependencies);
        *dep_iter_idx -= 1;
        *input_fields_out = std::move(input_fields);
        *res_fields_out = std::move(res_fields);
        *kernel_name_out = (*kernel_out)->kernel_name_;
        *dependent_iter_list_out = std::move(dependencies);
      } else if (func_name.compare(0, 20, "conditionedMergeJoin") == 0) {
        // FIXME
        int join_type = 0;
        gandiva::NodeVector left_schema_list;
        RETURN_NOT_OK(GetArguments(function_node, 0, &left_schema_list));
        gandiva::NodeVector right_schema_list;
        RETURN_NOT_OK(GetArguments(function_node, 1, &right_schema_list));
        gandiva::NodeVector left_key_list;
        RETURN_NOT_OK(GetArguments(function_node, 2, &left_key_list));
        gandiva::NodeVector right_key_list;
        RETURN_NOT_OK(GetArguments(function_node, 3, &right_key_list));
        gandiva::NodeVector result_list;
        RETURN_NOT_OK(GetArguments(function_node, 4, &result_list));
        gandiva::NodePtr condition;
        if (function_node->children().size() > 5) {
          condition = function_node->children()[5];
        }

        if (func_name.compare("conditionedMergeJoinInner") == 0) {
          join_type = 0;
        } else if (func_name.compare("conditionedMergeJoinOuter") == 0) {
          join_type = 1;
        } else if (func_name.compare("conditionedMergeJoinAnti") == 0) {
          join_type = 2;
        } else if (func_name.compare("conditionedMergeJoinSemi") == 0) {
          join_type = 3;
        } else if (func_name.compare("conditionedMergeJoinExistence") == 0) {
          join_type = 4;
        }
        std::vector<int> cur_hash_relation_idx;
        if (*hash_relation_idx == 0) {
          cur_hash_relation_idx = {*hash_relation_idx, *hash_relation_idx + 1};
          *hash_relation_idx += 2;
        } else {
          cur_hash_relation_idx = {*hash_relation_idx};
          *hash_relation_idx += 1;
        }
        RETURN_NOT_OK(ConditionedMergeJoinKernel::Make(
            ctx_, left_key_list, right_key_list, left_schema_list, right_schema_list,
            condition, join_type, result_list, cur_hash_relation_idx, kernel_out));
        is_smj_ = true;
        // treated as WSCG
        std::vector<std::shared_ptr<arrow::Field>> output_field_list;
        for (auto node : result_list) {
          output_field_list.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        std::vector<std::shared_ptr<arrow::Field>> left_key_field_list;
        std::vector<std::shared_ptr<arrow::Field>> left_field_list;
        for (auto node : left_key_list) {
          left_key_field_list.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        for (auto node : left_schema_list) {
          left_field_list.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        std::vector<std::shared_ptr<arrow::Field>> right_key_field_list;
        std::vector<std::shared_ptr<arrow::Field>> right_field_list;
        for (auto node : right_key_list) {
          right_key_field_list.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        for (auto node : right_schema_list) {
          right_field_list.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        THROW_NOT_OK(
            LoadJITFunction({}, output_field_list, {*kernel_out}, &wscg_handler_));
        auto res_schema = arrow::schema(output_field_list);
        wscg_handler_->MakeResultIterator(res_schema, gandiva_projector_list_, iter_out);
        auto left_iter = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
            dependent_iter_list[*dep_iter_idx]);
        *dep_iter_idx -= 1;
        auto right_iter = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
            dependent_iter_list[*dep_iter_idx]);
        *dep_iter_idx -= 1;
        left_iter->SetChildResIter(
            std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
                dependent_iter_list[*dep_iter_idx]));
        *dep_iter_idx -= 1;
        right_iter->SetChildResIter(
            std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
                dependent_iter_list[*dep_iter_idx]));
        *dep_iter_idx -= 1;

        std::shared_ptr<KernalBase> left_cache_kernel;
        std::shared_ptr<ResultIterator<SortRelation>> left_cache_iter;
        RETURN_NOT_OK(CachedRelationKernel::Make(ctx_, arrow::schema(left_field_list),
                                                 left_key_field_list, 0,
                                                 &left_cache_kernel));
        while (left_iter->HasNext()) {
          std::shared_ptr<arrow::RecordBatch> batch;
          RETURN_NOT_OK(left_iter->Next(&batch));
          auto cols = batch->columns();
          left_cache_kernel->Evaluate(cols);
        }
        left_cache_kernel->MakeResultIterator(arrow::schema(left_field_list),
                                              &left_cache_iter);
        std::shared_ptr<KernalBase> right_cache_kernel;
        std::shared_ptr<ResultIterator<SortRelation>> right_cache_iter;
        RETURN_NOT_OK(CachedRelationKernel::Make(ctx_, arrow::schema(right_field_list),
                                                 right_key_field_list, 0,
                                                 &right_cache_kernel));
        while (right_iter->HasNext()) {
          std::shared_ptr<arrow::RecordBatch> batch;
          RETURN_NOT_OK(right_iter->Next(&batch));
          auto cols = batch->columns();
          right_cache_kernel->Evaluate(cols);
        }
        right_cache_kernel->MakeResultIterator(arrow::schema(right_field_list),
                                               &right_cache_iter);

        std::vector<std::shared_ptr<ResultIteratorBase>> dependencies;
        dependencies.push_back(
            std::dynamic_pointer_cast<ResultIteratorBase>(left_cache_iter));
        dependencies.push_back(
            std::dynamic_pointer_cast<ResultIteratorBase>(right_cache_iter));
        // (*iter_out)->SetDependencies(dependencies);
        *res_fields_out = std::move(output_field_list);
        *kernel_name_out = (*kernel_out)->kernel_name_;
        *dependent_iter_list_out = std::move(dependencies);
      } else if (func_name.compare("CondProject") == 0) {
        // Will parse this node later because whether to do codegen needs to be decided.
        *function_node_out = std::move(function_node);
        *kernel_name_out = "CondProjectKernel";
      } else if (func_name.compare("project") == 0) {
        auto project_expression_list =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(function_node->children()[1])
                ->children();
        auto field_node_list =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(function_node->children()[0])
                ->children();
        RETURN_NOT_OK(ProjectKernel::Make(ctx_, field_node_list, project_expression_list,
                                          kernel_out));
        int idx = 0;
        std::vector<std::shared_ptr<arrow::Field>> input_fields;
        for (auto node : field_node_list) {
          input_fields.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        std::vector<std::shared_ptr<arrow::Field>> res_fields;
        for (auto node : project_expression_list) {
          auto res_field = arrow::field("projection_key_" + std::to_string(idx++),
                                        node->return_type());
          res_fields.push_back(res_field);
        }
        (*kernel_out)->MakeResultIterator(arrow::schema(res_fields), iter_out);
        *input_fields_out = std::move(input_fields);
        *res_fields_out = std::move(res_fields);
        *kernel_name_out = (*kernel_out)->kernel_name_;
      } else if (func_name.compare("filter") == 0) {
        auto field_node_list =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(function_node->children()[0])
                ->children();
        RETURN_NOT_OK(FilterKernel::Make(ctx_, field_node_list,
                                         function_node->children()[1], kernel_out));
        std::vector<std::shared_ptr<arrow::Field>> fields;
        for (auto node : field_node_list) {
          fields.push_back(std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        (*kernel_out)->MakeResultIterator(arrow::schema(fields), iter_out);
        *input_fields_out = std::move(fields);
        *res_fields_out = std::move(fields);
        *kernel_name_out = (*kernel_out)->kernel_name_;
      } else if (func_name.compare("hashAggregateArrays") == 0) {
        is_aggr_ = true;
        auto field_node_list =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(function_node->children()[0])
                ->children();
        auto action_node_list =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(function_node->children()[1])
                ->children();

        gandiva::NodeVector result_field_node_list;
        gandiva::NodeVector result_expr_node_list;
        if (function_node->children().size() == 4) {
          result_field_node_list = std::dynamic_pointer_cast<gandiva::FunctionNode>(
                                       function_node->children()[2])
                                       ->children();
          result_expr_node_list = std::dynamic_pointer_cast<gandiva::FunctionNode>(
                                      function_node->children()[3])
                                      ->children();
        }
        RETURN_NOT_OK(HashAggregateKernel::Make(ctx_, field_node_list, action_node_list,
                                                result_field_node_list,
                                                result_expr_node_list, kernel_out));
        std::vector<std::shared_ptr<arrow::Field>> input_fields;
        for (auto node : field_node_list) {
          input_fields.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        std::vector<std::shared_ptr<arrow::Field>> res_fields;
        for (auto node : result_field_node_list) {
          res_fields.push_back(
              std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
        }
        (*kernel_out)->MakeResultIterator(arrow::schema(res_fields), iter_out);
        *input_fields_out = std::move(input_fields);
        *res_fields_out = std::move(res_fields);
        *kernel_name_out = (*kernel_out)->kernel_name_;
      } else if (func_name.compare("sortArraysToIndices") == 0) {
        std::vector<std::shared_ptr<arrow::Field>> key_field_list;
        std::vector<bool> sort_directions;
        std::vector<bool> nulls_order;
        int result_type = 0;
        std::shared_ptr<arrow::Schema> result_schema;

        auto children = function_node->children();
        // first child is key_function
        auto sort_key_node_ =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(children[0])->children();
        // second child is key_field
        auto key_field_node =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(children[1]);
        for (auto field : key_field_node->children()) {
          auto field_node = std::dynamic_pointer_cast<gandiva::FieldNode>(field);
          key_field_list.push_back(field_node->field());
        }
        // third child is sort_directions
        auto sort_directions_node =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(children[2]);
        for (auto direction : sort_directions_node->children()) {
          auto dir_node = std::dynamic_pointer_cast<gandiva::LiteralNode>(direction);
          bool dir_val = arrow::util::get<bool>(dir_node->holder());
          sort_directions.push_back(dir_val);
        }
        // fourth child is nulls_order
        auto nulls_order_node =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(children[3]);
        for (auto order : nulls_order_node->children()) {
          auto order_node = std::dynamic_pointer_cast<gandiva::LiteralNode>(order);
          bool order_val = arrow::util::get<bool>(order_node->holder());
          nulls_order.push_back(order_val);
        }
        // fifth child specifies whether to check NaN when sorting
        auto nan_func_node =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(children[4]);
        auto NaN_lit_node =
            std::dynamic_pointer_cast<gandiva::LiteralNode>(nan_func_node->children()[0]);
        bool NaN_check_ = arrow::util::get<bool>(NaN_lit_node->holder());
        // sixth child specifies whether to do codegen for mutiple-key sort
        auto codegen_func_node =
            std::dynamic_pointer_cast<gandiva::FunctionNode>(children[5]);
        auto codegen_lit_node = std::dynamic_pointer_cast<gandiva::LiteralNode>(
            codegen_func_node->children()[0]);
        bool do_codegen_ = arrow::util::get<bool>(codegen_lit_node->holder());
        std::vector<std::shared_ptr<arrow::Field>> fields;
        if (children.size() == 7) {
          auto field_node_list =
              std::dynamic_pointer_cast<gandiva::FunctionNode>(children[6])->children();
          for (auto node : field_node_list) {
            fields.push_back(
                std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
          }
        }
        result_schema = arrow::schema(fields);
        RETURN_NOT_OK(SortArraysToIndicesKernel::Make(
            ctx_, result_schema, sort_key_node_, key_field_list, sort_directions,
            nulls_order, NaN_check_, do_codegen_, result_type, kernel_out));
        (*kernel_out)->MakeResultIterator(result_schema, iter_out);
        *input_fields_out = std::move(fields);
        *res_fields_out = std::move(fields);
        *kernel_name_out = (*kernel_out)->kernel_name_;
      } else {
        return arrow::Status::NotImplemented("Not supported function name:", func_name);
      }
      return arrow::Status::OK();
    }

    /* *
     * Expecting insert node is a function node whose function name is "child",
     * and real function is its first child, if who has two children, second one
     * is the next child.
     * */
    arrow::Status ParseNodeTree(
        std::shared_ptr<gandiva::Node> root_node,
        const std::vector<std::shared_ptr<ResultIteratorBase>>& dependent_iter_list,
        int* hash_relation_index, int* dep_iter_index,
        std::vector<std::shared_ptr<KernelInfo>>* kernel_info_list) {
      auto function_node = std::dynamic_pointer_cast<gandiva::FunctionNode>(root_node);
      if (function_node->descriptor()->name() != "child") {
        return arrow::Status::NotImplemented(
            "WholeStageTransformResultIterator expect child keyword.");
      }
      auto children = function_node->children();
      if (children.size() > 1) {
        ParseNodeTree(children[1], dependent_iter_list, hash_relation_index,
                      dep_iter_index, kernel_info_list);
      }
      std::shared_ptr<KernalBase> kernel;
      std::string kernel_name;
      std::vector<std::shared_ptr<arrow::Field>> input_fields;
      std::vector<std::shared_ptr<arrow::Field>> res_fields;
      std::shared_ptr<ResultIterator<arrow::RecordBatch>> res_iter;
      std::vector<std::shared_ptr<ResultIteratorBase>> kernel_dependent_iter_list;
      std::shared_ptr<gandiva::FunctionNode> child_node = nullptr;
      RETURN_NOT_OK(CreateKernelAndSetDep(
          children[0], dependent_iter_list, hash_relation_index, dep_iter_index,
          &kernel_name, &kernel, &res_iter, &input_fields, &res_fields,
          &kernel_dependent_iter_list, &child_node));
      auto kernel_info = std::make_shared<KernelInfo>(
          kernel, res_iter, input_fields, res_fields, kernel_name,
          kernel_dependent_iter_list, child_node);
      (*kernel_info_list).push_back(kernel_info);
      return arrow::Status::OK();
    }

    arrow::Status ParseCondProjectByCodegen(
        std::vector<std::shared_ptr<KernelInfo>>* kernel_info_list) {
      auto cond_project_name = "CondProjectKernel";
      std::vector<std::shared_ptr<KernelInfo>> new_kernel_info_list;

      for (auto kernel_info : (*kernel_info_list)) {
        if (kernel_info->kernel_name != cond_project_name) {
          new_kernel_info_list.push_back(kernel_info);
        } else {
          auto function_node = kernel_info->function_node;
          if (function_node == nullptr) {
            return arrow::Status::NotImplemented(
                "Please set function node for CondProject kernel info.");
          }
          int32_t children_size = function_node->children().size();
          if (children_size > 2) {
            return arrow::Status::NotImplemented(
                "Not supported more than two children in CondProject.");
          }
          gandiva::NodeVector filter_input_field_node_list;
          gandiva::NodeVector project_input_field_node_list;
          gandiva::NodePtr condition = nullptr;
          gandiva::NodeVector project_list = {};
          for (int i = 0; i < children_size; i++) {
            auto project_or_filter = std::dynamic_pointer_cast<gandiva::FunctionNode>(
                function_node->children()[i]);
            std::string node_name = project_or_filter->descriptor()->name();
            if (node_name == "project") {
              project_input_field_node_list =
                  std::dynamic_pointer_cast<gandiva::FunctionNode>(
                      project_or_filter->children()[0])
                      ->children();
              project_list = std::dynamic_pointer_cast<gandiva::FunctionNode>(
                                 project_or_filter->children()[1])
                                 ->children();
            } else if (node_name == "filter") {
              filter_input_field_node_list =
                  std::dynamic_pointer_cast<gandiva::FunctionNode>(
                      project_or_filter->children()[0])
                      ->children();
              condition = project_or_filter->children()[1];
            } else {
              return arrow::Status::NotImplemented("Not supported in CondProject:",
                                                   node_name);
            }
          }
          std::shared_ptr<KernalBase> kernel_out;
          std::shared_ptr<ResultIterator<arrow::RecordBatch>> iter_out;
          if (kernel_info->do_wscg) {
            // will create filter and project kernel separately for doing codegen
            if (condition != nullptr) {
              RETURN_NOT_OK(FilterKernel::Make(ctx_, filter_input_field_node_list,
                                               condition, &kernel_out));
              std::vector<std::shared_ptr<arrow::Field>> fields;
              for (auto node : filter_input_field_node_list) {
                fields.push_back(
                    std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
              }
              kernel_out->MakeResultIterator(arrow::schema(fields), &iter_out);
              auto kernel_info = std::make_shared<KernelInfo>(
                  kernel_out, iter_out, fields, fields, kernel_out->kernel_name_);
              kernel_info->setDoCodegen(true);
              new_kernel_info_list.push_back(kernel_info);
            }
            if (project_list.size() > 0) {
              RETURN_NOT_OK(ProjectKernel::Make(ctx_, project_input_field_node_list,
                                                project_list, &kernel_out));
              int idx = 0;
              std::vector<std::shared_ptr<arrow::Field>> input_fields;
              std::vector<std::shared_ptr<arrow::Field>> res_fields;
              for (auto node : project_list) {
                input_fields.push_back(
                    std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
                auto res_field = arrow::field("projection_key_" + std::to_string(idx++),
                                              node->return_type());
                res_fields.push_back(res_field);
              }
              kernel_out->MakeResultIterator(arrow::schema(res_fields), &iter_out);
              auto kernel_info =
                  std::make_shared<KernelInfo>(kernel_out, iter_out, input_fields,
                                               res_fields, kernel_out->kernel_name_);
              kernel_info->setDoCodegen(true);
              new_kernel_info_list.push_back(kernel_info);
            }
          } else {
            // Will create CondProjectKernel for direct computing.
            // When containing two children, expect filter will calculated before project.
            auto last_children = std::dynamic_pointer_cast<gandiva::FunctionNode>(
                                     function_node->children()[children_size - 1])
                                     ->children();
            auto input_field_node_list =
                std::dynamic_pointer_cast<gandiva::FunctionNode>(last_children[0])
                    ->children();
            RETURN_NOT_OK(CondProjectKernel::Make(ctx_, input_field_node_list, condition,
                                                  project_list, &kernel_out));

            std::vector<std::shared_ptr<arrow::Field>> input_fields;
            for (auto node : input_field_node_list) {
              input_fields.push_back(
                  std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
            }
            std::vector<std::shared_ptr<arrow::Field>> res_fields;
            if (project_list.size() == 0) {
              // If only filter is required.
              res_fields = input_fields;
            } else {
              int idx = 0;
              for (auto node : project_list) {
                auto field = arrow::field("projection_key_" + std::to_string(idx++),
                                          node->return_type());
                res_fields.push_back(field);
              }
            }
            kernel_out->MakeResultIterator(arrow::schema(res_fields), &iter_out);
            auto kernel_info = std::make_shared<KernelInfo>(
                kernel_out, iter_out, input_fields, res_fields, kernel_out->kernel_name_);
            new_kernel_info_list.push_back(kernel_info);
          }
        }
      }
      *kernel_info_list = std::move(new_kernel_info_list);
      return arrow::Status::OK();
    }

    ////////////////////// WSCG Related Functions //////////////////////

    arrow::Status LoadJITFunction(
        const std::vector<std::shared_ptr<arrow::Field>>& input_field_list,
        const std::vector<std::shared_ptr<arrow::Field>>& output_field_list,
        const std::vector<std::shared_ptr<KernalBase>>& kernel_list,
        std::shared_ptr<CodeGenBase>* out) {
      int argument_id = 0;
      int level = 0;
      std::vector<std::shared_ptr<CodeGenContext>> codegen_ctx_list;
      std::vector<std::pair<std::pair<std::string, std::string>, gandiva::DataTypePtr>>
          input_list;
      for (int i = 0; i < input_field_list.size(); i++) {
        auto name = "typed_in_col_" + std::to_string(i);
        auto type = input_field_list[i]->type();
        input_list.push_back(std::make_pair(std::make_pair(name, ""), type));
      }
      for (auto kernel : kernel_list) {
        std::shared_ptr<CodeGenContext> child_codegen_ctx;
        RETURN_NOT_OK(
            kernel->DoCodeGen(level++, input_list, &child_codegen_ctx, &argument_id));
        codegen_ctx_list.push_back(child_codegen_ctx);
        input_list.clear();
        for (auto pair : child_codegen_ctx->output_list) {
          input_list.push_back(pair);
        }
      }
      std::string codes;
      RETURN_NOT_OK(
          DoCodeGen(input_field_list, output_field_list, codegen_ctx_list, &codes));
      // generate dll signature
      std::stringstream signature_ss;
      signature_ss << std::hex << std::hash<std::string>{}(codes);
      signature_ = signature_ss.str();
      auto file_lock = FileSpinLock();
      auto status = LoadLibrary(signature_, ctx_, out);

      if (!status.ok()) {
        // process
        try {
          // compile codes
          arrow::Status s = CompileCodes(codes, signature_);
          s = LoadLibrary(signature_, ctx_, out);
        } catch (const std::runtime_error& error) {
          FileSpinUnLock(file_lock);
          throw error;
        }
      }
      FileSpinUnLock(file_lock);
      return arrow::Status::OK();
    }

    arrow::Status DoCodeGen(
        const std::vector<std::shared_ptr<arrow::Field>>& input_field_list,
        const std::vector<std::shared_ptr<arrow::Field>>& output_field_list,
        const std::vector<std::shared_ptr<CodeGenContext>>& codegen_ctx_list,
        std::string* codes) {
      std::stringstream codes_ss;
      std::string out_list;
      std::stringstream define_ss;
      codes_ss << BaseCodes() << std::endl;
      codes_ss << R"(#include "precompile/builder.h")" << std::endl;
      codes_ss << R"(#include "utils/macros.h")" << std::endl;
      std::vector<std::string> headers;
      for (auto codegen_ctx : codegen_ctx_list) {
        for (auto header : codegen_ctx->header_codes) {
          if (std::find(headers.begin(), headers.end(), header) == headers.end()) {
            headers.push_back(header);
          }
        }
        if (codegen_ctx->gandiva_projector)
          gandiva_projector_list_.push_back(codegen_ctx->gandiva_projector);
      }
      for (auto header : headers) {
        if (!header.empty()) {
          codes_ss << header << std::endl;
        }
      }

      if (is_aggr_) {
        for (auto codegen_ctx : codegen_ctx_list) {
          aggr_finish_condition_ += codegen_ctx->aggregate_finish_condition_codes;
        }
      }

      codes_ss << R"(
  using namespace sparkcolumnarplugin::precompile;
  class TypedWholeStageCodeGenImpl : public CodeGenBase {
  public:
    TypedWholeStageCodeGenImpl(arrow::compute::ExecContext *ctx) : ctx_(ctx) {}
    ~TypedWholeStageCodeGenImpl() {}

    arrow::Status MakeResultIterator(
        std::shared_ptr<arrow::Schema> schema,
        std::vector<std::shared_ptr<GandivaProjector>> gandiva_projector_list,
        std::shared_ptr<ResultIterator<arrow::RecordBatch>> *out) override {
      *out = std::make_shared<WholeStageCodeGenResultIterator>(ctx_, gandiva_projector_list, schema);
      return arrow::Status::OK();
    }

  private:
    arrow::compute::ExecContext* ctx_;
    class WholeStageCodeGenResultIterator : public ResultIterator<arrow::RecordBatch> {
    public:
      WholeStageCodeGenResultIterator(arrow::compute::ExecContext* ctx,
                                      std::vector<std::shared_ptr<GandivaProjector>> gandiva_projector_list,
                                      const std::shared_ptr<arrow::Schema>& result_schema)
          : ctx_(ctx), result_schema_(result_schema), gandiva_projector_list_(gandiva_projector_list) {)";
      if (!is_aggr_) {
        codes_ss << GetBuilderInitializeCodes(output_field_list) << std::endl;
      } else {
        for (auto codegen_ctx : codegen_ctx_list) {
          codes_ss << codegen_ctx->aggregate_prepare_codes << std::endl;
        }
      }
      codes_ss << "}" << std::endl;

      codes_ss << "arrow::Status GetMetrics(std::shared_ptr<Metrics>* out) override {"
               << std::endl;
      codes_ss << "auto metrics = std::make_shared<Metrics>(" << codegen_ctx_list.size()
               << ");" << std::endl;
      for (int i = 0; i < codegen_ctx_list.size(); i++) {
        auto out_length_name = "codegen_out_length_" + std::to_string(i);
        auto process_time_name = "process_time_" + std::to_string(i);
        codes_ss << "metrics->output_length[" << i << "] = " << out_length_name << ";"
                 << std::endl;
        codes_ss << "metrics->process_time[" << i << "] = " << process_time_name << ";"
                 << std::endl;
      }
      codes_ss << "*out = metrics;" << std::endl;
      codes_ss << "return arrow::Status::OK();" << std::endl;
      codes_ss << "}" << std::endl;

      codes_ss << R"(
      arrow::Status SetDependencies(
          const std::vector<std::shared_ptr<ResultIteratorBase>>& dependent_iter_list) {
        )";
      for (auto codegen_ctx : codegen_ctx_list) {
        codes_ss << codegen_ctx->relation_prepare_codes << std::endl;
      }
      codes_ss << R"(
        return arrow::Status::OK();
      }
  )" << std::endl;

      if (!is_aggr_ && !is_smj_) {
        codes_ss
            << R"(arrow::Status Process(const std::vector<std::shared_ptr<arrow::Array>>& in,
                            std::shared_ptr<arrow::RecordBatch>* out,
                            const std::shared_ptr<arrow::Array>& selection = nullptr)
          override {)"
            << std::endl;
      } else {
        if (is_aggr_ && !is_smj_) {
          codes_ss
              << R"(arrow::Status ProcessAndCacheOne(const std::vector<std::shared_ptr<arrow::Array>>& in,
                            const std::shared_ptr<arrow::Array>& selection = nullptr)
          override {)"
              << std::endl;
        } else if (is_smj_) {
          codes_ss << R"(bool HasNext() override { return !should_stop_; })" << std::endl;
          codes_ss << R"(arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out)
          override {)"
                   << std::endl;
          codes_ss << "uint64_t out_length = 0;" << std::endl;
          if (is_aggr_) {
            codes_ss << "int gp_idx = 0;" << std::endl;
            codes_ss << "if(!" << aggr_finish_condition_ << ") {";
          }
        }
        codes_ss << "  int i = 0;" << std::endl;
      }

      // convert input data to typed array
      for (int i = 0; i < input_field_list.size(); i++) {
        auto typed_array_name = "typed_in_" + std::to_string(i);
        codes_ss << "auto " << typed_array_name << " = std::make_shared<"
                 << GetTypeString(input_field_list[i]->type(), "Array") << ">(in[" << i
                 << "]);";
      }
      if (codegen_ctx_list.size() > 0) {
        codes_ss << codegen_ctx_list[0]->unsafe_row_prepare_codes << std::endl;
      }
      if (!is_smj_) {
        codes_ss << R"(
            uint64_t out_length = 0;
            auto length = typed_in_0->length();
            for (int i = 0; i < length; i++) {
      )" << std::endl;
      } else {
        codes_ss << "while (!should_stop_ && out_length < " << GetBatchSize() << ") {"
                 << std::endl;
      }
      // input preparation
      for (int i = 0; i < input_field_list.size(); i++) {
        auto typed_array_name = "typed_in_" + std::to_string(i);
        auto name = "typed_in_col_" + std::to_string(i);
        auto validity = name + "_validity";
        if (input_field_list[i]->type()->id() == arrow::Type::STRING) {
          define_ss << "bool " << validity << ";" << std::endl;
          define_ss << GetCTypeString(input_field_list[i]->type()) << " " << name << ";"
                    << std::endl;
          codes_ss << validity << " = " << typed_array_name
                   << "->IsNull(i) ? false : true;" << std::endl;
          codes_ss << "if (" << validity << ") {" << std::endl;
          codes_ss << name << " = " << typed_array_name << "->GetString(i);" << std::endl;
          codes_ss << "}" << std::endl;

        } else {
          define_ss << "bool " << validity << ";" << std::endl;
          define_ss << GetCTypeString(input_field_list[i]->type()) << " " << name << ";"
                    << std::endl;
          codes_ss << validity << " = " << typed_array_name
                   << "->IsNull(i) ? false : true;" << std::endl;
          codes_ss << "if (" << validity << ") {" << std::endl;

          codes_ss << name << " = " << typed_array_name << "->GetView(i);" << std::endl;
          codes_ss << "}" << std::endl;
        }
      }
      // paste children's codegen
      int codegen_ctx_idx = 0;
      for (auto codegen_ctx : codegen_ctx_list) {
        auto tmp_idx = codegen_ctx_idx;
        codegen_ctx_idx++;
        if (enable_time_metrics_) {
          codes_ss << "struct timespec start_" << tmp_idx << ", end_" << tmp_idx << ";"
                   << std::endl;
          codes_ss << "clock_gettime(CLOCK_MONOTONIC_COARSE, &start_" << tmp_idx << ");"
                   << std::endl;
        }
        codes_ss << codegen_ctx->prepare_codes << std::endl;
        if (codegen_ctx_idx < codegen_ctx_list.size()) {
          codes_ss << codegen_ctx_list[codegen_ctx_idx]->unsafe_row_prepare_codes
                   << std::endl;
        }
        codes_ss << codegen_ctx->process_codes << std::endl;
        if (codegen_ctx->aggregate_finish_condition_codes.empty())
          codes_ss << "codegen_out_length_" << tmp_idx << " += 1;" << std::endl;
        else
          aggr_out_length_idxs.push_back("codegen_out_length_" + std::to_string(tmp_idx));
      }

      codes_ss << GetProcessMaterializeCodes(codegen_ctx_list.back()) << std::endl;
      if (!is_aggr_) codes_ss << "out_length += 1;" << std::endl;
      for (int ctx_idx = codegen_ctx_list.size() - 1; ctx_idx >= 0; ctx_idx--) {
        auto codegen_ctx = codegen_ctx_list[ctx_idx];
        codes_ss << codegen_ctx->finish_codes << std::endl;
        if (enable_time_metrics_) {
          codes_ss << "clock_gettime(CLOCK_MONOTONIC_COARSE, &end_" << ctx_idx << ");"
                   << std::endl;
          codes_ss << "process_time_" << ctx_idx << " += TIME_NANO_DIFF(end_" << ctx_idx
                   << ", start_" << ctx_idx << ");" << std::endl;
        }
      }
      codes_ss << "} // end of for loop" << std::endl;
      if (is_aggr_ && !is_smj_) {
        codes_ss << "return arrow::Status::OK();" << std::endl;
        codes_ss << "} // End of ProcessAndCacheOne" << std::endl << std::endl;
        codes_ss << "bool HasNext() override { return !should_stop_; }" << std::endl;
        codes_ss << "arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* "
                    "out) override {"
                 << std::endl;
        codes_ss << "uint64_t out_length = 0;" << std::endl;
        codes_ss << "int gp_idx = 0;" << std::endl;
      } else if (is_aggr_ && is_smj_) {
        codes_ss << "} // end of if do_hash_aggr_finish_condition" << std::endl;
      } else {
        codes_ss << GetProcessFinishCodes(output_field_list) << std::endl;
      }
      std::stringstream output_arr_list_ss;
      if (is_aggr_) {
        int idx = 0;
        for (auto codegen_ctx : codegen_ctx_list) {
          codes_ss << codegen_ctx->aggregate_finish_codes << std::endl;
          if (!codegen_ctx->aggregate_finish_codes.empty())
            codes_ss << aggr_out_length_idxs[idx++] << " = " << aggr_finish_condition_
                     << "_offset;" << std::endl;
        }
        output_arr_list_ss << aggr_finish_condition_ << "_out";
      } else {
        output_arr_list_ss << "{" << GetProcessOutListCodes(output_field_list) << "}";
      }
      codes_ss << "*out = arrow::RecordBatch::Make(result_schema_, out_length, "
               << output_arr_list_ss.str() << ");" << std::endl;
      codes_ss << "return arrow::Status::OK();" << std::endl;
      codes_ss << "} // end of function" << std::endl;

      codes_ss << R"(
      private:
      arrow::compute::ExecContext* ctx_;
      bool should_stop_ = false;
      std::vector<std::shared_ptr<GandivaProjector>> gandiva_projector_list_;
      std::shared_ptr<arrow::Schema> result_schema_;)"
               << std::endl;

      codes_ss << define_ss.str();
      std::vector<std::string> unique_defines = GetUniqueDefineCodes(codegen_ctx_list);
      for (auto definition : unique_defines) {
        codes_ss << definition << std::endl;
      }

      if (!is_aggr_)
        codes_ss << GetBuilderDefinitionCodes(output_field_list) << std::endl;
      for (auto codegen_ctx : codegen_ctx_list) {
        for (auto func_codes : codegen_ctx->function_list) {
          codes_ss << func_codes << std::endl;
        }
      }

      codes_ss << "// Metrics" << std::endl;
      for (int i = 0; i < codegen_ctx_list.size(); i++) {
        codes_ss << "uint64_t codegen_out_length_" << i << " = 0;" << std::endl;
        codes_ss << "uint64_t process_time_" << i << " = 0;" << std::endl;
      }

      codes_ss << "};" << std::endl;
      codes_ss << "};" << std::endl;
      codes_ss << R"(
  extern "C" void MakeCodeGen(arrow::compute::ExecContext *ctx,
                              std::shared_ptr<CodeGenBase> *out) {
    *out = std::make_shared<TypedWholeStageCodeGenImpl>(ctx);
  })";

      *codes = codes_ss.str();
      return arrow::Status::OK();
    }

    std::string GetProcessMaterializeCodes(std::shared_ptr<CodeGenContext> codegen_ctx) {
      std::stringstream codes_ss;
      int i = 0;
      auto out_list = codegen_ctx->output_list;
      for (int j = 0; j < out_list.size(); j++) {
        auto pair = out_list[j];
        auto name = pair.first.first;
        auto type = pair.second;
        auto validity = name + "_validity";
        codes_ss << pair.first.second << std::endl;
        codes_ss << "if (" << validity << ") {" << std::endl;
        if (type->id() == arrow::Type::STRING) {
          codes_ss << "  RETURN_NOT_OK(builder_" << i << "_->AppendString(" << name
                   << "));" << std::endl;
        } else {
          codes_ss << "  RETURN_NOT_OK(builder_" << i << "_->Append(" << name << "));"
                   << std::endl;
        }
        codes_ss << "} else {" << std::endl;
        codes_ss << "  RETURN_NOT_OK(builder_" << i << "_->AppendNull());" << std::endl;
        codes_ss << "}" << std::endl;
        i++;
      }
      return codes_ss.str();
    }

    std::string GetProcessFinishCodes(gandiva::FieldVector output_field_list) {
      std::stringstream codes_ss;
      for (int i = 0; i < output_field_list.size(); i++) {
        auto data_type = output_field_list[i]->type();
        codes_ss << "std::shared_ptr<arrow::Array> out_" << i << ";" << std::endl;
        codes_ss << "RETURN_NOT_OK(builder_" << i << "_->Finish(&out_" << i << "));"
                 << std::endl;
        codes_ss << "builder_" << i << "_->Reset();" << std::endl;
      }
      return codes_ss.str();
    }

    std::string GetProcessOutListCodes(gandiva::FieldVector output_field_list) {
      std::vector<std::string> output_list;
      for (int i = 0; i < output_field_list.size(); i++) {
        output_list.push_back("out_" + std::to_string(i));
      }
      std::stringstream codes_ss;
      codes_ss << GetParameterList(output_list, false);
      return codes_ss.str();
    }

    std::string GetBuilderInitializeCodes(gandiva::FieldVector output_field_list) {
      std::stringstream codes_ss;
      for (int i = 0; i < output_field_list.size(); i++) {
        auto data_type = output_field_list[i]->type();
        if (data_type->id() == arrow::Type::DECIMAL ||
            data_type->id() == arrow::Type::TIMESTAMP) {
          codes_ss << "builder_" << i << "_ = std::make_shared<"
                   << GetTypeString(data_type, "Builder")
                   << ">(arrow::" << GetArrowTypeDefString(data_type)
                   << ", ctx_->memory_pool());" << std::endl;
        } else {
          codes_ss << "builder_" << i << "_ = std::make_shared<"
                   << GetTypeString(data_type, "Builder") << ">(ctx_->memory_pool());"
                   << std::endl;
        }
      }
      return codes_ss.str();
    }

    std::string GetBuilderDefinitionCodes(gandiva::FieldVector output_field_list) {
      std::stringstream codes_ss;
      for (int i = 0; i < output_field_list.size(); i++) {
        auto data_type = output_field_list[i]->type();
        codes_ss << "std::shared_ptr<" << GetTypeString(data_type, "Builder")
                 << "> builder_" << i << "_;" << std::endl;
      }
      return codes_ss.str();
    }

    // This function is used to find the unique definitions,
    // by dividing the definition_codes with line breaks.
    std::vector<std::string> GetUniqueDefineCodes(
        const std::vector<std::shared_ptr<CodeGenContext>>& codegen_ctx_list) {
      std::vector<std::string> unique_defines;
      std::string delimiter = "\n";
      for (auto codegen_ctx : codegen_ctx_list) {
        std::string define_codes = codegen_ctx->definition_codes;
        int pos = 0;
        std::string definition;
        while ((pos = define_codes.find(delimiter)) != std::string::npos) {
          definition = define_codes.substr(0, pos) + delimiter;
          if (std::find(unique_defines.begin(), unique_defines.end(), definition) ==
              unique_defines.end()) {
            unique_defines.push_back(definition);
          }
          define_codes.erase(0, pos + delimiter.length());
        }
      }
      return unique_defines;
    }
  };
};

arrow::Status WholeStageTransformKernel::Make(
    arrow::compute::ExecContext* ctx,
    const std::vector<std::shared_ptr<arrow::Field>>& input_field_list,
    std::shared_ptr<gandiva::Node> root_node,
    const std::vector<std::shared_ptr<arrow::Field>>& output_field_list,
    std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<WholeStageTransformKernel>(ctx, input_field_list, root_node,
                                                     output_field_list);
  return arrow::Status::OK();
}

WholeStageTransformKernel::WholeStageTransformKernel(
    arrow::compute::ExecContext* ctx,
    const std::vector<std::shared_ptr<arrow::Field>>& input_field_list,
    std::shared_ptr<gandiva::Node> root_node,
    const std::vector<std::shared_ptr<arrow::Field>>& output_field_list) {
  impl_.reset(new Impl(ctx, input_field_list, root_node, output_field_list));
  kernel_name_ = "WholeStageTransformKernel";
  ctx_ = nullptr;
}

arrow::Status WholeStageTransformKernel::MakeResultIterator(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
  return impl_->MakeResultIterator(schema, out);
}

std::string WholeStageTransformKernel::GetSignature() { return impl_->GetSignature(); }

}  // namespace extra
}  // namespace arrowcompute
}  // namespace codegen
}  // namespace sparkcolumnarplugin
