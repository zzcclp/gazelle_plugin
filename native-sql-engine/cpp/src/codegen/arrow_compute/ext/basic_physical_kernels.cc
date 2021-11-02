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
#include <gandiva/filter.h>
#include <gandiva/node.h>
#include <gandiva/projector.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "codegen/arrow_compute/ext/codegen_common.h"
#include "codegen/arrow_compute/ext/expression_codegen_visitor.h"
#include "codegen/arrow_compute/ext/kernels_ext.h"
#include "codegen/common/hash_relation.h"
#include "utils/macros.h"

namespace sparkcolumnarplugin {
namespace codegen {
namespace arrowcompute {
namespace extra {

using ArrayList = std::vector<std::shared_ptr<arrow::Array>>;

///////////////  Project  ////////////////
class ProjectKernel::Impl {
 public:
  Impl(arrow::compute::ExecContext* ctx, const gandiva::NodeVector& input_field_node_list,
       const gandiva::NodeVector& project_list)
      : ctx_(ctx), project_list_(project_list) {
    for (auto node : input_field_node_list) {
      auto field_node = std::dynamic_pointer_cast<gandiva::FieldNode>(node);
      input_field_list_.push_back(field_node->field());
    }
    pool_ = nullptr;
  }

  arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
    auto input_schema = arrow::schema(input_field_list_);
    *out = std::make_shared<ProjectResultIterator>(ctx_, project_list_, input_schema,
                                                   schema);
    return arrow::Status::OK();
  }

  std::string GetSignature() { return signature_; }

  arrow::Status DoCodeGen(
      int level,
      std::vector<std::pair<std::pair<std::string, std::string>, gandiva::DataTypePtr>>
          input,
      std::shared_ptr<CodeGenContext>* codegen_ctx_out, int* var_id) {
    auto codegen_ctx = std::make_shared<CodeGenContext>();
    int idx = 0;
    for (auto project : project_list_) {
      std::shared_ptr<ExpressionCodegenVisitor> project_node_visitor;
      std::vector<std::string> input_list;
      std::vector<int> indices_list;
      auto is_local = false;
      RETURN_NOT_OK(MakeExpressionCodegenVisitor(project, &input, {input_field_list_}, -1,
                                                 var_id, is_local, &input_list,
                                                 &project_node_visitor));
      codegen_ctx->process_codes += project_node_visitor->GetPrepare();
      auto name = project_node_visitor->GetResult();
      auto validity = project_node_visitor->GetPreCheck();

      auto output_name =
          "project_" + std::to_string(level) + "_output_col_" + std::to_string(idx++);
      auto output_validity = output_name + "_validity";
      std::stringstream output_get_ss;
      output_get_ss << output_name << " = " << name << ";" << std::endl;
      output_get_ss << output_validity << " = " << validity << ";" << std::endl;
      codegen_ctx->output_list.push_back(std::make_pair(
          std::make_pair(output_name, output_get_ss.str()), project->return_type()));
      std::stringstream value_define_ss;
      value_define_ss << "bool " << output_validity << ";" << std::endl;
      value_define_ss << project_node_visitor->GetResType() << " " << output_name << ";"
                      << std::endl;
      codegen_ctx->definition_codes += value_define_ss.str();
      for (auto header : project_node_visitor->GetHeaders()) {
        if (std::find(codegen_ctx->header_codes.begin(), codegen_ctx->header_codes.end(),
                      header) == codegen_ctx->header_codes.end()) {
          codegen_ctx->header_codes.push_back(header);
        }
      }
    }
    *codegen_ctx_out = codegen_ctx;
    return arrow::Status::OK();
  }

  class ProjectResultIterator : public ResultIterator<arrow::RecordBatch> {
   public:
    ProjectResultIterator(arrow::compute::ExecContext* ctx,
                          const gandiva::NodeVector& project_list,
                          const std::shared_ptr<arrow::Schema>& input_schema,
                          const std::shared_ptr<arrow::Schema>& result_schema)
        : ctx_(ctx),
          project_list_(project_list),
          input_schema_(input_schema),
          result_schema_(result_schema) {
      pool_ = ctx_->memory_pool();
      gandiva::ExpressionVector project_exprs = GetGandivaKernel(project_list);
      auto configuration = gandiva::ConfigurationBuilder().DefaultConfiguration();
      THROW_NOT_OK(gandiva::Projector::Make(input_schema, project_exprs, configuration,
                                            &key_projector_));
    }

    arrow::Status SetChildResIter(
        const std::shared_ptr<ResultIterator<arrow::RecordBatch>>& iter) override {
      child_iter_ = iter;
      return arrow::Status::OK();
    }

    bool HasNext() override {
      while (num_rows_ == 0) {
        if (!child_iter_->HasNext()) {
          return false;
        }
        THROW_NOT_OK(child_iter_->Next(&next_batch_));
        num_rows_ += next_batch_->num_rows();
      }
      return true;
    }

    arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out) override {
      arrow::ArrayVector outputs;
      RETURN_NOT_OK(key_projector_->Evaluate(*next_batch_, pool_, &outputs));
      auto length = outputs.size() > 0 ? outputs[0]->length() : 0;
      *out = arrow::RecordBatch::Make(result_schema_, length, outputs);
      num_rows_ = 0;
      return arrow::Status::OK();
    }

   private:
    arrow::compute::ExecContext* ctx_;
    arrow::MemoryPool* pool_;
    gandiva::NodeVector project_list_;
    std::shared_ptr<ResultIterator<arrow::RecordBatch>> child_iter_;
    std::shared_ptr<gandiva::Projector> key_projector_;
    std::shared_ptr<arrow::Schema> input_schema_;
    std::shared_ptr<arrow::Schema> result_schema_;
    std::shared_ptr<arrow::RecordBatch> next_batch_;
    int64_t num_rows_ = 0;
  };

 private:
  arrow::compute::ExecContext* ctx_;
  arrow::MemoryPool* pool_;
  std::string signature_;
  gandiva::NodeVector project_list_;
  gandiva::FieldVector input_field_list_;
};

arrow::Status ProjectKernel::Make(arrow::compute::ExecContext* ctx,
                                  const gandiva::NodeVector& input_field_node_list,
                                  const gandiva::NodeVector& project_list,
                                  std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<ProjectKernel>(ctx, input_field_node_list, project_list);
  return arrow::Status::OK();
}

ProjectKernel::ProjectKernel(arrow::compute::ExecContext* ctx,
                             const gandiva::NodeVector& input_field_node_list,
                             const gandiva::NodeVector& project_list) {
  impl_.reset(new Impl(ctx, input_field_node_list, project_list));
  kernel_name_ = "ProjectKernel";
  ctx_ = nullptr;
}

arrow::Status ProjectKernel::MakeResultIterator(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
  return impl_->MakeResultIterator(schema, out);
}

std::string ProjectKernel::GetSignature() { return impl_->GetSignature(); }

arrow::Status ProjectKernel::DoCodeGen(
    int level,
    std::vector<std::pair<std::pair<std::string, std::string>, gandiva::DataTypePtr>>
        input,
    std::shared_ptr<CodeGenContext>* codegen_ctx, int* var_id) {
  return impl_->DoCodeGen(level, input, codegen_ctx, var_id);
}

///////////////  Filter  ////////////////
class FilterKernel::Impl {
 public:
  Impl(arrow::compute::ExecContext* ctx, const gandiva::NodeVector& input_field_node_list,
       const gandiva::NodePtr& condition)
      : ctx_(ctx), condition_(condition), input_field_node_list_(input_field_node_list) {
    for (auto node : input_field_node_list) {
      auto field_node = std::dynamic_pointer_cast<gandiva::FieldNode>(node);
      input_field_list_.push_back(field_node->field());
    }
    pool_ = nullptr;
  }

  arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
    *out = std::make_shared<FilterResultIterator>(
        ctx_, condition_, input_field_node_list_, input_field_list_);

    return arrow::Status::OK();
  }

  std::string GetSignature() { return signature_; }

  arrow::Status DoCodeGen(
      int level,
      std::vector<std::pair<std::pair<std::string, std::string>, gandiva::DataTypePtr>>
          input,
      std::shared_ptr<CodeGenContext>* codegen_ctx_out, int* var_id) {
    auto codegen_ctx = std::make_shared<CodeGenContext>();
    std::shared_ptr<ExpressionCodegenVisitor> condition_node_visitor;
    std::vector<std::string> input_list;
    std::vector<int> indices_list;
    auto is_local = false;
    RETURN_NOT_OK(MakeExpressionCodegenVisitor(condition_, &input, {input_field_list_},
                                               -1, var_id, is_local, &input_list,
                                               &condition_node_visitor));
    codegen_ctx->process_codes += condition_node_visitor->GetPrepare();
    for (auto header : condition_node_visitor->GetHeaders()) {
      if (!header.empty() &&
          std::find(codegen_ctx->header_codes.begin(), codegen_ctx->header_codes.end(),
                    header) == codegen_ctx->header_codes.end()) {
        codegen_ctx->header_codes.push_back(header);
      }
    }

    auto condition_codes = condition_node_visitor->GetResult();
    std::stringstream process_ss;
    process_ss << "if (!(" << condition_codes << ")) {" << std::endl;
    process_ss << "continue;" << std::endl;
    process_ss << "}" << std::endl;
    int idx = 0;
    for (auto field : input_field_list_) {
      codegen_ctx->output_list.push_back(
          std::make_pair(std::make_pair(input[idx].first.first, input[idx].first.second),
                         field->type()));
      idx++;
    }
    codegen_ctx->process_codes += process_ss.str();

    *codegen_ctx_out = codegen_ctx;

    return arrow::Status::OK();
  }

  class FilterResultIterator : public ResultIterator<arrow::RecordBatch> {
   public:
    FilterResultIterator(arrow::compute::ExecContext* ctx,
                         const gandiva::NodePtr& condition_node,
                         const gandiva::NodeVector& input_field_node_list,
                         const gandiva::FieldVector input_field_list)
        : ctx_(ctx),
          input_field_list_(input_field_list),
          res_field_list_(input_field_list) {
      pool_ = ctx_->memory_pool();
      auto condition = gandiva::TreeExprBuilder::MakeCondition(condition_node);
      auto configuration = gandiva::ConfigurationBuilder().DefaultConfiguration();
      auto input_schema = arrow::schema(input_field_list);
      THROW_NOT_OK(
          gandiva::Filter::Make(input_schema, condition, configuration, &filter_));
      gandiva::ExpressionVector project_exprs =
          GetGandivaKernelWithResField(input_field_node_list, res_field_list_);
      THROW_NOT_OK(gandiva::Projector::Make(input_schema, project_exprs,
                                            gandiva::SelectionVector::MODE_UINT32,
                                            configuration, &projector_));
    }

    arrow::Status SetChildResIter(
        const std::shared_ptr<ResultIterator<arrow::RecordBatch>>& iter) override {
      child_iter_ = iter;
      return arrow::Status::OK();
    }

    bool HasNext() override {
      while (num_rows_ == 0) {
        if (!child_iter_->HasNext()) {
          return false;
        }
        std::shared_ptr<arrow::RecordBatch> next_batch;
        THROW_NOT_OK(child_iter_->Next(&next_batch));
        std::shared_ptr<gandiva::SelectionVector> selection_vector;
        THROW_NOT_OK(gandiva::SelectionVector::MakeInt32(next_batch->num_rows(), pool_,
                                                         &selection_vector));
        THROW_NOT_OK(filter_->Evaluate(*next_batch, selection_vector));
        num_rows_ += selection_vector->GetNumSlots();
        if (num_rows_ > 0) {
          THROW_NOT_OK(projector_->Evaluate(*next_batch, selection_vector.get(), pool_,
                                            &outputs_));
        }
      }
      return true;
    }

    arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out) override {
      *out =
          arrow::RecordBatch::Make(arrow::schema(res_field_list_), num_rows_, outputs_);
      num_rows_ = 0;
      return arrow::Status::OK();
    }

   private:
    arrow::compute::ExecContext* ctx_;
    arrow::MemoryPool* pool_;
    std::shared_ptr<ResultIterator<arrow::RecordBatch>> child_iter_;
    gandiva::FieldVector input_field_list_;
    arrow::FieldVector res_field_list_;
    std::shared_ptr<gandiva::Filter> filter_;
    std::shared_ptr<gandiva::Projector> projector_;
    arrow::ArrayVector outputs_;
    int64_t num_rows_ = 0;
  };

 private:
  arrow::compute::ExecContext* ctx_;
  arrow::MemoryPool* pool_;
  std::string signature_;
  gandiva::NodePtr condition_;
  gandiva::FieldVector input_field_list_;
  gandiva::NodeVector input_field_node_list_;
};

arrow::Status FilterKernel::Make(arrow::compute::ExecContext* ctx,
                                 const gandiva::NodeVector& input_field_node_list,
                                 const gandiva::NodePtr& condition,
                                 std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<FilterKernel>(ctx, input_field_node_list, condition);
  return arrow::Status::OK();
}

FilterKernel::FilterKernel(arrow::compute::ExecContext* ctx,
                           const gandiva::NodeVector& input_field_node_list,
                           const gandiva::NodePtr& condition) {
  impl_.reset(new Impl(ctx, input_field_node_list, condition));
  kernel_name_ = "FilterKernel";
}

arrow::Status FilterKernel::MakeResultIterator(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
  return impl_->MakeResultIterator(schema, out);
}

std::string FilterKernel::GetSignature() { return impl_->GetSignature(); }

arrow::Status FilterKernel::DoCodeGen(
    int level,
    std::vector<std::pair<std::pair<std::string, std::string>, gandiva::DataTypePtr>>
        input,
    std::shared_ptr<CodeGenContext>* codegen_ctx, int* var_id) {
  return impl_->DoCodeGen(level, input, codegen_ctx, var_id);
}

///////////////  CondProject  ////////////////
class CondProjectKernel::Impl {
 public:
  Impl(arrow::compute::ExecContext* ctx, const gandiva::NodeVector& input_field_node_list,
       const gandiva::NodePtr& condition, const gandiva::NodeVector& project_list)
      : ctx_(ctx),
        input_field_node_list_(input_field_node_list),
        condition_(condition),
        project_list_(project_list) {
    for (auto node : input_field_node_list) {
      auto field_node = std::dynamic_pointer_cast<gandiva::FieldNode>(node);
      input_field_list_.push_back(field_node->field());
    }
    pool_ = nullptr;
  }

  arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
    *out = std::make_shared<CondProjectResultIterator>(ctx_, input_field_node_list_,
                                                       input_field_list_, schema,
                                                       condition_, project_list_);
    return arrow::Status::OK();
  }

  std::string GetSignature() { return signature_; }

  class CondProjectResultIterator : public ResultIterator<arrow::RecordBatch> {
   public:
    CondProjectResultIterator(arrow::compute::ExecContext* ctx,
                              const gandiva::NodeVector& input_field_node_list,
                              const gandiva::FieldVector& input_field_list,
                              const std::shared_ptr<arrow::Schema>& result_schema,
                              const gandiva::NodePtr& condition_node,
                              const gandiva::NodeVector& project_list)
        : ctx_(ctx), input_field_list_(input_field_list), result_schema_(result_schema) {
      pool_ = ctx_->memory_pool();
      auto configuration = gandiva::ConfigurationBuilder().DefaultConfiguration();
      auto input_schema = arrow::schema(input_field_list);
      if (condition_node != nullptr) {
        auto condition = gandiva::TreeExprBuilder::MakeCondition(condition_node);
        THROW_NOT_OK(
            gandiva::Filter::Make(input_schema, condition, configuration, &filter_));
      }
      if (project_list.size() != 0) {
        gandiva::ExpressionVector project_exprs = GetGandivaKernel(project_list);
        if (filter_) {
          THROW_NOT_OK(gandiva::Projector::Make(input_schema, project_exprs,
                                                gandiva::SelectionVector::MODE_UINT32,
                                                configuration, &projector_));
        } else {
          THROW_NOT_OK(gandiva::Projector::Make(input_schema, project_exprs,
                                                configuration, &projector_));
        }
      } else {
        gandiva::ExpressionVector project_exprs =
            GetGandivaKernelWithResField(input_field_node_list, input_field_list_);
        THROW_NOT_OK(gandiva::Projector::Make(input_schema, project_exprs,
                                              gandiva::SelectionVector::MODE_UINT32,
                                              configuration, &projector_));
      }
    }

    arrow::Status SetChildResIter(
        const std::shared_ptr<ResultIterator<arrow::RecordBatch>>& iter) {
      child_iter_ = iter;
      return arrow::Status::OK();
    }

    bool HasNext() override {
      while (num_rows_ == 0) {
        if (!child_iter_->HasNext()) {
          return false;
        }
        THROW_NOT_OK(child_iter_->Next(&next_batch_));
        if (filter_ == nullptr) {
          num_rows_ += next_batch_->num_rows();
          if (num_rows_ > 0) {
            THROW_NOT_OK(projector_->Evaluate(*next_batch_, pool_, &outputs_));
          }
        } else {
          std::shared_ptr<gandiva::SelectionVector> selection_vector;
          THROW_NOT_OK(gandiva::SelectionVector::MakeInt32(next_batch_->num_rows(), pool_,
                                                           &selection_vector));
          THROW_NOT_OK(filter_->Evaluate(*next_batch_, selection_vector));
          num_rows_ += selection_vector->GetNumSlots();
          if (num_rows_ > 0) {
            THROW_NOT_OK(projector_->Evaluate(*next_batch_, selection_vector.get(), pool_,
                                              &outputs_));
          }
        }
      }
      return true;
    }

    arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out) override {
      *out = arrow::RecordBatch::Make(result_schema_, num_rows_, outputs_);
      num_rows_ = 0;
      return arrow::Status::OK();
    }

   private:
    arrow::compute::ExecContext* ctx_;
    arrow::MemoryPool* pool_;
    gandiva::FieldVector input_field_list_;
    std::shared_ptr<arrow::Schema> result_schema_;
    std::shared_ptr<ResultIterator<arrow::RecordBatch>> child_iter_;
    gandiva::NodeVector project_list_;
    std::shared_ptr<gandiva::Filter> filter_ = nullptr;
    std::shared_ptr<gandiva::Projector> projector_ = nullptr;
    std::shared_ptr<arrow::RecordBatch> next_batch_;
    arrow::ArrayVector outputs_;
    int64_t num_rows_ = 0;
  };

 private:
  arrow::compute::ExecContext* ctx_;
  arrow::MemoryPool* pool_;
  gandiva::NodeVector input_field_node_list_;
  gandiva::FieldVector input_field_list_;
  std::string signature_;
  gandiva::NodePtr condition_;
  gandiva::NodeVector project_list_;
};

arrow::Status CondProjectKernel::Make(arrow::compute::ExecContext* ctx,
                                      const gandiva::NodeVector& input_field_node_list,
                                      const gandiva::NodePtr& condition,
                                      const gandiva::NodeVector& project_list,
                                      std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<CondProjectKernel>(ctx, input_field_node_list, condition,
                                             project_list);
  return arrow::Status::OK();
}

CondProjectKernel::CondProjectKernel(arrow::compute::ExecContext* ctx,
                                     const gandiva::NodeVector& input_field_node_list,
                                     const gandiva::NodePtr& condition,
                                     const gandiva::NodeVector& project_list) {
  impl_.reset(new Impl(ctx, input_field_node_list, condition, project_list));
  kernel_name_ = "CondProjectKernel";
  ctx_ = nullptr;
}

arrow::Status CondProjectKernel::MakeResultIterator(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
  return impl_->MakeResultIterator(schema, out);
}

std::string CondProjectKernel::GetSignature() { return impl_->GetSignature(); }

}  // namespace extra
}  // namespace arrowcompute
}  // namespace codegen
}  // namespace sparkcolumnarplugin