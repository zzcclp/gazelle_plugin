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

#pragma once
#include <arrow/compute/api.h>
#include <arrow/type.h>
#include <gandiva/node.h>
#include <gandiva/tree_expr_builder.h>

#include <sstream>
#include <string>

#include "codegen/arrow_compute/ext/code_generator_base.h"
#include "codegen/arrow_compute/ext/kernels_ext.h"

namespace sparkcolumnarplugin {
namespace codegen {
namespace arrowcompute {
namespace extra {

std::string BaseCodes();

int FileSpinLock();

void FileSpinUnLock(int fd);

int GetBatchSize();
bool GetEnableTimeMetrics();
std::string exec(const char* cmd);
std::string GetTempPath();
std::string GetArrowTypeDefString(std::shared_ptr<arrow::DataType> type);
std::string GetCTypeString(std::shared_ptr<arrow::DataType> type);
std::string GetTypeString(std::shared_ptr<arrow::DataType> type,
                          std::string tail = "Type");
std::string GetTemplateString(std::shared_ptr<arrow::DataType> type,
                              std::string template_name, std::string tail = "",
                              std::string prefix = "");
bool StrCmpCaseInsensitive(const std::string& str1, const std::string& str2);
gandiva::ExpressionPtr GetConcatedKernel(std::vector<gandiva::NodePtr> key_list);
gandiva::ExpressionPtr GetHash32Kernel(std::vector<gandiva::NodePtr> key_list);
gandiva::ExpressionPtr GetHash32Kernel(std::vector<gandiva::NodePtr> key_list,
                                       std::vector<int> key_index_list);
gandiva::ExpressionVector GetGandivaKernel(std::vector<gandiva::NodePtr> key_list);
gandiva::ExpressionVector GetGandivaKernelWithResField(
    std::vector<gandiva::NodePtr> key_list, arrow::FieldVector res_field_list);
template <typename T>
std::string GetStringFromList(std::vector<T> list) {
  std::stringstream ss;
  for (auto i : list) {
    ss << i << std::endl;
  }
  return ss.str();
}
std::string GetParameterList(std::vector<std::string> parameter_list,
                             bool comma_ahead = true, std::string split = ", ");
arrow::Status GetIndexList(const std::vector<std::shared_ptr<arrow::Field>>& target_list,
                           const std::vector<std::shared_ptr<arrow::Field>>& source_list,
                           std::vector<int>* out);
arrow::Status GetIndexList(
    const std::vector<std::shared_ptr<arrow::Field>>& target_list,
    const std::vector<std::shared_ptr<arrow::Field>>& left_field_list,
    const std::vector<std::shared_ptr<arrow::Field>>& right_field_list,
    const bool isExistJoin, int* exist_index,
    std::vector<std::pair<int, int>>* result_schema_index_list);
std::vector<int> GetIndicesFromSchemaCaseInsensitive(
    const std::shared_ptr<arrow::Schema>& result_schema, const std::string& field_name);
arrow::Status GetIndexListFromSchema(
    const std::shared_ptr<arrow::Schema>& result_schema,
    const std::vector<std::shared_ptr<arrow::Field>>& field_list,
    std::vector<int>* index_list);
std::pair<int, int> GetFieldIndex(gandiva::FieldPtr target_field,
                                  std::vector<gandiva::FieldVector> field_list_v);

arrow::Status CompileCodes(std::string codes, std::string signature);

arrow::Status LoadLibrary(std::string signature, arrow::compute::ExecContext* ctx,
                          std::shared_ptr<CodeGenBase>* out);

/* This struct is used for conducting WSCG in ws transform */
struct KernelInfo {
  KernelInfo(
      const std::shared_ptr<KernalBase>& kernel,
      const std::shared_ptr<ResultIterator<arrow::RecordBatch>>& res_iter,
      const std::vector<std::shared_ptr<arrow::Field>>& input_fields,
      const std::vector<std::shared_ptr<arrow::Field>>& res_fields,
      const std::string& kernel_name,
      const std::vector<std::shared_ptr<ResultIteratorBase>>& dependent_iter_list = {},
      const std::shared_ptr<gandiva::FunctionNode>& function_node = nullptr)
      : kernel(kernel),
        kernel_name(kernel_name),
        input_fields(input_fields),
        res_fields(res_fields),
        res_iter(res_iter),
        dependent_iter_list(dependent_iter_list),
        function_node(function_node) {}

  void setDoCodegen(bool will_do_wscg) { do_wscg = will_do_wscg; }

  std::string kernel_name;
  std::shared_ptr<KernalBase> kernel;
  std::vector<std::shared_ptr<arrow::Field>> input_fields;
  std::vector<std::shared_ptr<arrow::Field>> res_fields;
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> res_iter;
  std::vector<std::shared_ptr<ResultIteratorBase>> dependent_iter_list;
  std::shared_ptr<gandiva::FunctionNode> function_node;
  bool do_wscg = false;
};

}  // namespace extra
}  // namespace arrowcompute
}  // namespace codegen
}  // namespace sparkcolumnarplugin
