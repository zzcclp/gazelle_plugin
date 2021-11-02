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

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/pretty_print.h>
#include <arrow/record_batch.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/type.h>
#include <arrow/util/decimal.h>
#include <arrow/util/range.h>
#include <dirent.h>
#include <gandiva/basic_decimal_scalar.h>
#include <gandiva/gandiva_aliases.h>
#include <gandiva/node.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <stdio.h>
#include <sys/types.h>

#include <chrono>

#include "codegen/code_generator.h"
#include "codegen/code_generator_factory.h"
#include "codegen/common/result_iterator.h"
#include "tests/test_utils.h"

using arrow::Decimal128;
using gandiva::DecimalScalar128;

namespace sparkcolumnarplugin {
namespace codegen {

class SimulatedQueryTest : public ::testing::Test {
 public:
  void SetUp() override { SetUpDataSourceDir("/home/morui/Pictures/lineitem_parquets/"); }

  bool EndsWith(const std::string& data, const std::string& suffix) {
    return data.find(suffix, data.size() - suffix.size()) != std::string::npos;
  }

  std::vector<std::string> ListFile(std::string file_dir) {
    const char* path = const_cast<char*>(file_dir.c_str());
    struct dirent* entry;
    DIR* dir = opendir(path);
    std::vector<std::string> file_list;
    if (dir == NULL) {
      return file_list;
    }
    const std::string prefix = "file://";
    int64_t num_file = 0;
    while ((entry = readdir(dir)) != NULL) {
      std::string file{entry->d_name};
      if (EndsWith(file, ".parquet")) {
        file_list.push_back(prefix + file_dir + file);
        std::cout << "file: " << file << "num: " << num_file << std::endl;
        num_file += 1;
      }
    }
    closedir(dir);
    return file_list;
  }

  void SetUpDataSourceDir(std::string file_dir) {
    std::vector<std::string> file_list = ListFile(file_dir);
    if (file_list.size() == 0) {
      throw std::runtime_error("no file");
    }
    std::shared_ptr<DataSourceKernel> ds_kernel =
        std::make_shared<DataSourceKernel>(file_list);
    field_list_.push_back(ds_kernel->getFieldList());
    std::shared_ptr<ResultIterator<arrow::RecordBatch>> ds_iter;
    ds_kernel->MakeResultIterator(&ds_iter);
    std::shared_ptr<ResultIteratorBase> base_iter =
        std::dynamic_pointer_cast<ResultIteratorBase>(ds_iter);
    ds_kernel_list_.push_back(ds_kernel);
    ds_iter_list_.push_back(base_iter);
  }

  class DataSourceKernel {
   public:
    DataSourceKernel(const std::vector<std::string>& file_list) {
      // read input from multiple parquet files
      for (auto single_file : file_list) {
        std::string path = single_file;
        std::string file_name;
        std::shared_ptr<arrow::fs::FileSystem> fs =
            arrow::fs::FileSystemFromUri(path, &file_name).ValueOrDie();
        std::shared_ptr<arrow::io::RandomAccessFile> file;
        ARROW_ASSIGN_OR_THROW(file, fs->OpenInputFile(file_name));
        file_list_.push_back(file);

        parquet::ReaderProperties reader_properties =
            parquet::default_reader_properties();
        reader_properties.set_buffer_size(20480);
        std::unique_ptr<parquet::ParquetFileReader> reader =
            parquet::ParquetFileReader::OpenFile(file_name, true, reader_properties);

        std::unique_ptr<::parquet::arrow::FileReader> parquet_reader;
        auto pool = arrow::default_memory_pool();
        ::parquet::arrow::FileReader::Make(pool, std::move(reader), &parquet_reader);
        std::vector<int> all_row_groups = arrow::internal::Iota(
            parquet_reader->parquet_reader()->metadata()->num_row_groups());

        std::shared_ptr<RecordBatchReader> record_batch_reader;
        ASSERT_NOT_OK(parquet_reader->GetRecordBatchReader(all_row_groups, {4, 5, 6, 10},
                                                           &record_batch_reader));

        std::shared_ptr<::parquet::arrow::FileReader> parquet_reader_shared =
            std::move(parquet_reader);
        parquet_reader_list_.push_back(parquet_reader_shared);
        record_batch_reader_list_.push_back(record_batch_reader);
      }
      if (record_batch_reader_list_.size() == 0) {
        throw std::runtime_error("No valid file!");
      }
      ////////////////// expr prepration ////////////////
      field_list_ = record_batch_reader_list_[0]->schema()->fields();
    }

    arrow::Status MakeResultIterator(
        std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
      *out = std::make_shared<DataSourceResultIterator>(record_batch_reader_list_);
      return arrow::Status::OK();
    }

    std::vector<std::shared_ptr<::arrow::Field>> getFieldList() {
      std::cout << "Datasource input fields: " << std::endl;
      for (int i = 0; i < field_list_.size(); i++) {
        std::cout << field_list_[i]->ToString() << std::endl;
      }
      return field_list_;
    }

   private:
    std::vector<std::shared_ptr<::arrow::Field>> field_list_;
    std::vector<std::shared_ptr<arrow::io::RandomAccessFile>> file_list_;
    std::vector<std::shared_ptr<::parquet::arrow::FileReader>> parquet_reader_list_;
    std::vector<std::shared_ptr<RecordBatchReader>> record_batch_reader_list_;

    class DataSourceResultIterator : public ResultIterator<arrow::RecordBatch> {
     public:
      DataSourceResultIterator(
          const std::vector<std::shared_ptr<RecordBatchReader>>& record_batch_reader_list)
          : record_batch_reader_list_(record_batch_reader_list) {}

      bool HasNext() override {
        while (current_reader_idx_ < record_batch_reader_list_.size()) {
          record_batch_reader_ = record_batch_reader_list_[current_reader_idx_];
          record_batch_reader_->ReadNext(&record_batch_);
          if (record_batch_) {
            // arrow::PrettyPrint(*record_batch_.get(), 2, &std::cout);
            return true;
          } else {
            current_reader_idx_ += 1;
          }
        }
        return false;
      }

      arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out) override {
        *out = record_batch_;
        return arrow::Status::OK();
      }

     private:
      std::shared_ptr<RecordBatchReader> record_batch_reader_;
      std::vector<std::shared_ptr<RecordBatchReader>> record_batch_reader_list_;
      std::shared_ptr<arrow::RecordBatch> record_batch_;
      int64_t current_reader_idx_ = 0;
    };
  };

  void StartWithIterator(std::shared_ptr<CodeGenerator> ws_transform_expr,
                         const std::vector<std::shared_ptr<ResultIteratorBase>>& deps) {
    std::shared_ptr<arrow::RecordBatch> record_batch;
    std::shared_ptr<ResultIteratorBase> base_result_iterator;
    ASSERT_NOT_OK(ws_transform_expr->finish(&base_result_iterator));
    auto typed_result_iterator =
        std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
            base_result_iterator);
    typed_result_iterator->SetDependencies(deps);
    uint64_t num_output_batches = 0;
    std::shared_ptr<arrow::RecordBatch> result_batch;
    while (typed_result_iterator->HasNext()) {
      typed_result_iterator->Next(&result_batch);
      arrow::PrettyPrint(*result_batch.get(), 2, &std::cout);
      num_output_batches++;
    }
    std::cout << "num_output_batches: " << num_output_batches << std::endl;
  }

 protected:
  std::vector<std::vector<std::shared_ptr<::arrow::Field>>> field_list_;
  std::vector<std::shared_ptr<::arrow::Field>> ret_field_list_;
  std::shared_ptr<arrow::Field> f_res;
  std::vector<std::shared_ptr<DataSourceKernel>> ds_kernel_list_;
  std::vector<std::shared_ptr<ResultIteratorBase>> ds_iter_list_;

  uint64_t num_batches = 0;
};

TEST_F(SimulatedQueryTest, SimulatedQ6Test) {
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> gandiva_field_list;
  for (auto field : field_list_[0]) {
    gandiva_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  // Project
  auto n_project_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_project_func = TreeExprBuilder::MakeFunction(
      "codegen_project", {gandiva_field_list[1], gandiva_field_list[2]}, uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  // Filter
  auto n_and_0 = TreeExprBuilder::MakeAnd(
      {TreeExprBuilder::MakeFunction("isnotnull", {gandiva_field_list[3]},
                                     arrow::boolean()),
       TreeExprBuilder::MakeFunction("isnotnull", {gandiva_field_list[2]},
                                     arrow::boolean())});
  auto n_and_1 = TreeExprBuilder::MakeAnd(
      {n_and_0, TreeExprBuilder::MakeFunction("isnotnull", {gandiva_field_list[0]},
                                              arrow::boolean())});

  auto n_greater_than_or_equal_to_0 = TreeExprBuilder::MakeFunction(
      "greater_than_or_equal_to",
      {gandiva_field_list[3], TreeExprBuilder::MakeLiteral((double)8766)},
      arrow::boolean());
  auto n_and_2 = TreeExprBuilder::MakeAnd({n_and_1, n_greater_than_or_equal_to_0});

  auto n_less_than_0 = TreeExprBuilder::MakeFunction(
      "less_than", {gandiva_field_list[3], TreeExprBuilder::MakeLiteral((double)9131)},
      arrow::boolean());
  auto n_and_3 = TreeExprBuilder::MakeAnd({n_and_2, n_less_than_0});

  auto n_greater_than_or_equal_to_1 = TreeExprBuilder::MakeFunction(
      "greater_than_or_equal_to",
      {gandiva_field_list[2], TreeExprBuilder::MakeLiteral((double)0.05)},
      arrow::boolean());
  auto n_and_4 = TreeExprBuilder::MakeAnd({n_and_3, n_greater_than_or_equal_to_1});

  auto n_less_than_or_equal_to_0 = TreeExprBuilder::MakeFunction(
      "less_than_or_equal_to",
      {gandiva_field_list[2], TreeExprBuilder::MakeLiteral((double)0.07)},
      arrow::boolean());
  auto n_and_5 = TreeExprBuilder::MakeAnd({n_and_4, n_less_than_or_equal_to_0});

  auto n_less_than_1 = TreeExprBuilder::MakeFunction(
      "less_than", {gandiva_field_list[0], TreeExprBuilder::MakeLiteral((double)24.00)},
      arrow::boolean());
  auto n_and_6 = TreeExprBuilder::MakeAnd({n_and_5, n_less_than_1});

  auto n_filter_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_filter =
      TreeExprBuilder::MakeFunction("filter", {n_filter_input, n_and_6}, uint32());
  // CondProject
  auto n_cond_project =
      TreeExprBuilder::MakeFunction("CondProject", {n_project, n_filter}, uint32());
  auto n_cond_project_child =
      TreeExprBuilder::MakeFunction("child", {n_cond_project}, uint32());

  // Aggregate
  auto n_mul_0 = TreeExprBuilder::MakeFunction(
      "multiply", {gandiva_field_list[1], gandiva_field_list[2]}, arrow::float64());
  auto n_sum_0 = TreeExprBuilder::MakeFunction("action_sum_partial", {n_mul_0}, uint32());

  auto n_action = TreeExprBuilder::MakeFunction("aggregateActions", {n_sum_0}, uint32());
  // res fields
  auto f_sum_0 = field("sum_0", arrow::float64());

  // aggregate expressions
  auto n_proj = TreeExprBuilder::MakeFunction(
      "aggregateExpressions", {gandiva_field_list[1], gandiva_field_list[2]}, uint32());

  std::vector<std::shared_ptr<::gandiva::Node>> agg_res_fields_list = {
      TreeExprBuilder::MakeField(f_sum_0)};
  auto n_result =
      TreeExprBuilder::MakeFunction("resultSchema", agg_res_fields_list, uint32());
  auto n_result_expr =
      TreeExprBuilder::MakeFunction("resultExpressions", agg_res_fields_list, uint32());
  auto n_aggr = TreeExprBuilder::MakeFunction(
      "hashAggregateArrays", {n_proj, n_action, n_result, n_result_expr}, uint32());

  auto n_child =
      TreeExprBuilder::MakeFunction("child", {n_aggr, n_cond_project_child}, uint32());
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());

  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);

  std::shared_ptr<arrow::Schema> schema;
  schema = arrow::schema(field_list_[0]);

  ret_field_list_ = {f_sum_0};
  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

TEST_F(SimulatedQueryTest, SimulatedQ6MergeProjectTest) {
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> gandiva_field_list;
  for (auto field : field_list_[0]) {
    gandiva_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  // Project
  auto n_project_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_mul_0 = TreeExprBuilder::MakeFunction(
      "multiply", {gandiva_field_list[1], gandiva_field_list[2]}, arrow::float64());
  auto n_project_func =
      TreeExprBuilder::MakeFunction("codegen_project", {n_mul_0}, uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  // Filter
  auto n_and_0 = TreeExprBuilder::MakeAnd(
      {TreeExprBuilder::MakeFunction("isnotnull", {gandiva_field_list[3]},
                                     arrow::boolean()),
       TreeExprBuilder::MakeFunction("isnotnull", {gandiva_field_list[2]},
                                     arrow::boolean())});
  auto n_and_1 = TreeExprBuilder::MakeAnd(
      {n_and_0, TreeExprBuilder::MakeFunction("isnotnull", {gandiva_field_list[0]},
                                              arrow::boolean())});

  auto n_greater_than_or_equal_to_0 = TreeExprBuilder::MakeFunction(
      "greater_than_or_equal_to",
      {gandiva_field_list[3], TreeExprBuilder::MakeLiteral((double)8766)},
      arrow::boolean());
  auto n_and_2 = TreeExprBuilder::MakeAnd({n_and_1, n_greater_than_or_equal_to_0});

  auto n_less_than_0 = TreeExprBuilder::MakeFunction(
      "less_than", {gandiva_field_list[3], TreeExprBuilder::MakeLiteral((double)9131)},
      arrow::boolean());
  auto n_and_3 = TreeExprBuilder::MakeAnd({n_and_2, n_less_than_0});

  auto n_greater_than_or_equal_to_1 = TreeExprBuilder::MakeFunction(
      "greater_than_or_equal_to",
      {gandiva_field_list[2], TreeExprBuilder::MakeLiteral((double)0.05)},
      arrow::boolean());
  auto n_and_4 = TreeExprBuilder::MakeAnd({n_and_3, n_greater_than_or_equal_to_1});

  auto n_less_than_or_equal_to_0 = TreeExprBuilder::MakeFunction(
      "less_than_or_equal_to",
      {gandiva_field_list[2], TreeExprBuilder::MakeLiteral((double)0.07)},
      arrow::boolean());
  auto n_and_5 = TreeExprBuilder::MakeAnd({n_and_4, n_less_than_or_equal_to_0});

  auto n_less_than_1 = TreeExprBuilder::MakeFunction(
      "less_than", {gandiva_field_list[0], TreeExprBuilder::MakeLiteral((double)24.00)},
      arrow::boolean());
  auto n_and_6 = TreeExprBuilder::MakeAnd({n_and_5, n_less_than_1});

  auto n_filter_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_filter =
      TreeExprBuilder::MakeFunction("filter", {n_filter_input, n_and_6}, uint32());
  // CondProject
  auto n_cond_project =
      TreeExprBuilder::MakeFunction("CondProject", {n_project, n_filter}, uint32());
  auto n_cond_project_child =
      TreeExprBuilder::MakeFunction("child", {n_cond_project}, uint32());

  // Aggregate
  auto f_mul_0 = field("mul_0", arrow::float64());
  auto n_sum_0 = TreeExprBuilder::MakeFunction(
      "action_sum_partial", {TreeExprBuilder::MakeField(f_mul_0)}, uint32());

  auto n_action = TreeExprBuilder::MakeFunction("aggregateActions", {n_sum_0}, uint32());
  // res fields
  auto f_sum_0 = field("sum_0", arrow::float64());

  // aggregate expressions
  auto n_proj = TreeExprBuilder::MakeFunction(
      "aggregateExpressions", {TreeExprBuilder::MakeField(f_mul_0)}, uint32());

  std::vector<std::shared_ptr<::gandiva::Node>> agg_res_fields_list = {
      TreeExprBuilder::MakeField(f_sum_0)};
  auto n_result =
      TreeExprBuilder::MakeFunction("resultSchema", agg_res_fields_list, uint32());
  auto n_result_expr =
      TreeExprBuilder::MakeFunction("resultExpressions", agg_res_fields_list, uint32());
  auto n_aggr = TreeExprBuilder::MakeFunction(
      "hashAggregateArrays", {n_proj, n_action, n_result, n_result_expr}, uint32());

  auto n_child =
      TreeExprBuilder::MakeFunction("child", {n_aggr, n_cond_project_child}, uint32());
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());

  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);

  std::shared_ptr<arrow::Schema> schema;
  schema = arrow::schema(field_list_[0]);

  ret_field_list_ = {f_sum_0};
  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

TEST_F(SimulatedQueryTest, SimulatedQ6NewCondtionTest) {
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> gandiva_field_list;
  for (auto field : field_list_[0]) {
    gandiva_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  // Project
  auto n_project_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_mul_0 = TreeExprBuilder::MakeFunction(
      "multiply", {gandiva_field_list[1], gandiva_field_list[2]}, arrow::float64());
  auto n_project_func =
      TreeExprBuilder::MakeFunction("codegen_project", {n_mul_0}, uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  // Filter
  auto n_and_0 = TreeExprBuilder::MakeAnd(
      {TreeExprBuilder::MakeFunction("isnotnull", {gandiva_field_list[3]},
                                     arrow::boolean()),
       TreeExprBuilder::MakeFunction("isnotnull", {gandiva_field_list[2]},
                                     arrow::boolean())});
  auto n_and_1 = TreeExprBuilder::MakeAnd(
      {n_and_0, TreeExprBuilder::MakeFunction("isnotnull", {gandiva_field_list[0]},
                                              arrow::boolean())});

  auto n_greater_than_or_equal_to_0 = TreeExprBuilder::MakeFunction(
      "greater_than_or_equal_to",
      {gandiva_field_list[3], TreeExprBuilder::MakeLiteral((double)0.0)},
      arrow::boolean());
  auto n_and_2 = TreeExprBuilder::MakeAnd({n_and_1, n_greater_than_or_equal_to_0});

  auto n_less_than_0 = TreeExprBuilder::MakeFunction(
      "less_than",
      {gandiva_field_list[3], TreeExprBuilder::MakeLiteral((double)10000000.0)},
      arrow::boolean());
  auto n_and_3 = TreeExprBuilder::MakeAnd({n_and_2, n_less_than_0});

  auto n_greater_than_or_equal_to_1 = TreeExprBuilder::MakeFunction(
      "greater_than_or_equal_to",
      {gandiva_field_list[2], TreeExprBuilder::MakeLiteral((double)0.0)},
      arrow::boolean());
  auto n_and_4 = TreeExprBuilder::MakeAnd({n_and_3, n_greater_than_or_equal_to_1});

  auto n_less_than_or_equal_to_0 = TreeExprBuilder::MakeFunction(
      "less_than_or_equal_to",
      {gandiva_field_list[2], TreeExprBuilder::MakeLiteral((double)1.0)},
      arrow::boolean());
  auto n_and_5 = TreeExprBuilder::MakeAnd({n_and_4, n_less_than_or_equal_to_0});

  auto n_less_than_1 = TreeExprBuilder::MakeFunction(
      "less_than",
      {gandiva_field_list[0], TreeExprBuilder::MakeLiteral((double)10000000.0)},
      arrow::boolean());
  auto n_and_6 = TreeExprBuilder::MakeAnd({n_and_5, n_less_than_1});

  auto n_filter_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_filter =
      TreeExprBuilder::MakeFunction("filter", {n_filter_input, n_and_6}, uint32());
  // CondProject
  auto n_cond_project =
      TreeExprBuilder::MakeFunction("CondProject", {n_project, n_filter}, uint32());
  auto n_cond_project_child =
      TreeExprBuilder::MakeFunction("child", {n_cond_project}, uint32());

  // Aggregate
  auto f_mul_0 = field("mul_0", arrow::float64());
  auto n_sum_0 = TreeExprBuilder::MakeFunction(
      "action_sum_partial", {TreeExprBuilder::MakeField(f_mul_0)}, uint32());

  auto n_action = TreeExprBuilder::MakeFunction("aggregateActions", {n_sum_0}, uint32());
  // res fields
  auto f_sum_0 = field("sum_0", arrow::float64());

  // aggregate expressions
  auto n_proj = TreeExprBuilder::MakeFunction(
      "aggregateExpressions", {TreeExprBuilder::MakeField(f_mul_0)}, uint32());

  std::vector<std::shared_ptr<::gandiva::Node>> agg_res_fields_list = {
      TreeExprBuilder::MakeField(f_sum_0)};
  auto n_result =
      TreeExprBuilder::MakeFunction("resultSchema", agg_res_fields_list, uint32());
  auto n_result_expr =
      TreeExprBuilder::MakeFunction("resultExpressions", agg_res_fields_list, uint32());
  auto n_aggr = TreeExprBuilder::MakeFunction(
      "hashAggregateArrays", {n_proj, n_action, n_result, n_result_expr}, uint32());

  auto n_child =
      TreeExprBuilder::MakeFunction("child", {n_aggr, n_cond_project_child}, uint32());
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());

  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);

  std::shared_ptr<arrow::Schema> schema;
  schema = arrow::schema(field_list_[0]);

  ret_field_list_ = {f_sum_0};
  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

}  // namespace codegen
}  // namespace sparkcolumnarplugin
