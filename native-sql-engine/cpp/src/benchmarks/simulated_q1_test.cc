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
#include <gandiva/basic_decimal_scalar.h>
#include <gandiva/gandiva_aliases.h>
#include <gandiva/node.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

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
  void SetUp() override { SetUpDataSource("tpch_sf800_lineitem.parquet"); }

  void SetUpDataSource(std::string name) {
    std::shared_ptr<DataSourceKernel> ds_kernel =
        std::make_shared<DataSourceKernel>(name);
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
    DataSourceKernel(std::string name) {
      // read input from parquet file
#ifdef BENCHMARK_FILE_PATH
      std::string dir_path = BENCHMARK_FILE_PATH;
#else
      std::string dir_path = "";
#endif
      std::string path = dir_path + name;

      std::string file_name;
      std::shared_ptr<arrow::fs::FileSystem> fs =
          arrow::fs::FileSystemFromUri(path, &file_name).ValueOrDie();
      ARROW_ASSIGN_OR_THROW(file_, fs->OpenInputFile(file_name));

      parquet::ReaderProperties reader_properties = parquet::default_reader_properties();
      reader_properties.set_buffer_size(20480);
      std::unique_ptr<parquet::ParquetFileReader> reader =
          parquet::ParquetFileReader::OpenFile(file_name, true, reader_properties);

      auto pool = arrow::default_memory_pool();
      ::parquet::arrow::FileReader::Make(pool, std::move(reader), &parquet_reader_);
      std::vector<int> all_row_groups = arrow::internal::Iota(
          parquet_reader_->parquet_reader()->metadata()->num_row_groups());
      ASSERT_NOT_OK(parquet_reader_->GetRecordBatchReader(
          all_row_groups, {4, 5, 6, 7, 8, 9, 10}, &record_batch_reader_));

      ////////////////// expr prepration ////////////////
      field_list_ = record_batch_reader_->schema()->fields();
    }

    arrow::Status MakeResultIterator(
        std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
      *out = std::make_shared<DataSourceResultIterator>(record_batch_reader_);
      return arrow::Status::OK();
    }

    std::shared_ptr<::parquet::arrow::FileReader> getParquerReader() {
      return std::move(parquet_reader_);
    }

    std::vector<std::shared_ptr<::arrow::Field>> getFieldList() {
      std::cout << "Datasource input fields: " << std::endl;
      for (int i = 0; i < field_list_.size(); i++) {
        std::cout << field_list_[i]->ToString() << std::endl;
      }
      return field_list_;
    }

    std::shared_ptr<RecordBatchReader> getBatchReader() { return record_batch_reader_; }

   private:
    std::shared_ptr<arrow::io::RandomAccessFile> file_;
    std::unique_ptr<::parquet::arrow::FileReader> parquet_reader_;
    std::vector<std::shared_ptr<::arrow::Field>> field_list_;
    std::shared_ptr<RecordBatchReader> record_batch_reader_;

    class DataSourceResultIterator : public ResultIterator<arrow::RecordBatch> {
     public:
      DataSourceResultIterator(
          const std::shared_ptr<RecordBatchReader> record_batch_reader)
          : record_batch_reader_(record_batch_reader) {}

      bool HasNext() override {
        record_batch_reader_->ReadNext(&record_batch_);
        if (record_batch_) {
          return true;
        } else {
          return false;
        }
      }

      arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out) override {
        *out = record_batch_;
        return arrow::Status::OK();
      }

     private:
      std::shared_ptr<RecordBatchReader> record_batch_reader_;
      std::shared_ptr<arrow::RecordBatch> record_batch_;
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
      // arrow::PrettyPrint(*result_batch.get(), 2, &std::cout);
      // num_output_batches++;
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

TEST_F(SimulatedQueryTest, SimulatedQ1Test) {
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
      "codegen_project",
      {gandiva_field_list[0], gandiva_field_list[1], gandiva_field_list[2],
       gandiva_field_list[3], gandiva_field_list[4], gandiva_field_list[5]},
      uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  // Filter
  auto n_filter_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());

  auto n_date = TreeExprBuilder::MakeFunction(
      "castDATE", {TreeExprBuilder::MakeLiteral((int32_t)10471)}, arrow::date32());

  auto n_filter_func = TreeExprBuilder::MakeFunction(
      "less_than_or_equal_to", {gandiva_field_list[6], n_date}, arrow::boolean());
  auto n_filter =
      TreeExprBuilder::MakeFunction("filter", {n_filter_input, n_filter_func}, uint32());
  // CondProject
  auto n_cond_project =
      TreeExprBuilder::MakeFunction("CondProject", {n_project, n_filter}, uint32());
  auto n_cond_project_child =
      TreeExprBuilder::MakeFunction("child", {n_cond_project}, uint32());

  // Aggregate
  auto n_groupby_0 =
      TreeExprBuilder::MakeFunction("action_groupby", {gandiva_field_list[4]}, uint32());
  auto n_groupby_1 =
      TreeExprBuilder::MakeFunction("action_groupby", {gandiva_field_list[5]}, uint32());
  auto n_sum_0 = TreeExprBuilder::MakeFunction("action_sum_partial",
                                               {gandiva_field_list[0]}, uint32());
  auto n_sum_1 = TreeExprBuilder::MakeFunction("action_sum_partial",
                                               {gandiva_field_list[1]}, uint32());
  /// agg action
  auto n_sub_0 = TreeExprBuilder::MakeFunction(
      "subtract",
      {TreeExprBuilder::MakeDecimalLiteral(DecimalScalar128("1.00", 13, 2)),
       gandiva_field_list[2]},
      decimal128(14, 2));
  auto n_cast_dec_0 = TreeExprBuilder::MakeFunction("castDECIMALNullOnOverflow",
                                                    {n_sub_0}, decimal128(13, 2));
  auto n_mul_0 = TreeExprBuilder::MakeFunction(
      "multiply", {gandiva_field_list[1], n_cast_dec_0}, decimal128(26, 4));
  auto n_sum_2 = TreeExprBuilder::MakeFunction("action_sum_partial", {n_mul_0}, uint32());
  /// agg action
  auto n_add_0 = TreeExprBuilder::MakeFunction(
      "add",
      {TreeExprBuilder::MakeDecimalLiteral(DecimalScalar128("1.00", 13, 2)),
       gandiva_field_list[3]},
      decimal128(14, 2));
  auto n_cast_dec_1 = TreeExprBuilder::MakeFunction("castDECIMALNullOnOverflow",
                                                    {n_add_0}, decimal128(13, 2));
  auto n_mul_1 = TreeExprBuilder::MakeFunction("multiply", {n_mul_0, n_cast_dec_1},
                                               decimal128(38, 6));
  auto n_sum_3 = TreeExprBuilder::MakeFunction("action_sum_partial", {n_mul_1}, uint32());
  ///
  auto n_sum_count_0 = TreeExprBuilder::MakeFunction("action_sum_count",
                                                     {gandiva_field_list[0]}, uint32());
  auto n_sum_count_1 = TreeExprBuilder::MakeFunction("action_sum_count",
                                                     {gandiva_field_list[1]}, uint32());
  auto n_sum_count_2 = TreeExprBuilder::MakeFunction("action_sum_count",
                                                     {gandiva_field_list[2]}, uint32());
  auto n_count_0 = TreeExprBuilder::MakeFunction("action_countLiteral_1", {}, uint32());
  auto n_action = TreeExprBuilder::MakeFunction(
      "aggregateActions",
      {n_groupby_0, n_groupby_1, n_sum_0, n_sum_1, n_sum_2, n_sum_3, n_sum_count_0,
       n_sum_count_1, n_sum_count_2, n_count_0},
      uint32());
  // res fields
  auto f_sum_0 = field("sum_0", decimal128(22, 2));
  auto f_isEmpty_0 = field("isEmpty_0", boolean());
  auto f_sum_1 = field("sum_1", decimal128(22, 2));
  auto f_isEmpty_1 = field("isEmpty_1", boolean());
  auto f_sum_2 = field("sum_2", decimal128(36, 4));
  auto f_isEmpty_2 = field("isEmpty_2", boolean());
  auto f_sum_3 = field("sum_3", decimal128(38, 6));
  auto f_isEmpty_3 = field("isEmpty_3", boolean());
  auto f_sum_4 = field("sum_4", decimal128(22, 2));
  auto f_count_0 = field("count_0", int64());
  auto f_sum_5 = field("sum_5", decimal128(22, 2));
  auto f_count_1 = field("count_1", int64());
  auto f_sum_6 = field("sum_6", decimal128(22, 2));
  auto f_count_2 = field("count_2", int64());
  auto f_count_3 = field("count_3", int64());

  // aggregate expressions
  auto n_proj = TreeExprBuilder::MakeFunction(
      "aggregateExpressions",
      {gandiva_field_list[0], gandiva_field_list[1], gandiva_field_list[2],
       gandiva_field_list[3], gandiva_field_list[4], gandiva_field_list[5]},
      uint32());

  std::vector<std::shared_ptr<::gandiva::Node>> agg_res_fields_list = {
      gandiva_field_list[4],
      gandiva_field_list[5],
      TreeExprBuilder::MakeField(f_sum_0),
      TreeExprBuilder::MakeField(f_isEmpty_0),
      TreeExprBuilder::MakeField(f_sum_1),
      TreeExprBuilder::MakeField(f_isEmpty_1),
      TreeExprBuilder::MakeField(f_sum_2),
      TreeExprBuilder::MakeField(f_isEmpty_2),
      TreeExprBuilder::MakeField(f_sum_3),
      TreeExprBuilder::MakeField(f_isEmpty_3),
      TreeExprBuilder::MakeField(f_sum_4),
      TreeExprBuilder::MakeField(f_count_0),
      TreeExprBuilder::MakeField(f_sum_5),
      TreeExprBuilder::MakeField(f_count_1),
      TreeExprBuilder::MakeField(f_sum_6),
      TreeExprBuilder::MakeField(f_count_2),
      TreeExprBuilder::MakeField(f_count_3)};
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

  auto f4_name = field_list_[0][4]->name();
  auto f5_name = field_list_[0][5]->name();
  auto f4_type = field_list_[0][4]->type();
  auto f5_type = field_list_[0][5]->type();

  ret_field_list_ = {field(f4_name, f4_type),
                     field(f5_name, f5_type),
                     f_sum_0,
                     f_isEmpty_0,
                     f_sum_1,
                     f_isEmpty_1,
                     f_sum_2,
                     f_isEmpty_2,
                     f_sum_3,
                     f_isEmpty_3,
                     f_sum_4,
                     f_count_0,
                     f_sum_5,
                     f_count_1,
                     f_sum_6,
                     f_count_2,
                     f_count_3};
  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

TEST_F(SimulatedQueryTest, SimulatedQ1MergeProjectTest) {
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
  ///
  auto n_sub_0 = TreeExprBuilder::MakeFunction(
      "subtract",
      {TreeExprBuilder::MakeDecimalLiteral(DecimalScalar128("1.00", 13, 2)),
       gandiva_field_list[2]},
      decimal128(14, 2));
  auto n_cast_dec_0 = TreeExprBuilder::MakeFunction("castDECIMALNullOnOverflow",
                                                    {n_sub_0}, decimal128(13, 2));
  auto n_mul_0 = TreeExprBuilder::MakeFunction(
      "multiply", {gandiva_field_list[1], n_cast_dec_0}, decimal128(26, 4));
  ///
  auto n_add_0 = TreeExprBuilder::MakeFunction(
      "add",
      {TreeExprBuilder::MakeDecimalLiteral(DecimalScalar128("1.00", 13, 2)),
       gandiva_field_list[3]},
      decimal128(14, 2));
  auto n_cast_dec_1 = TreeExprBuilder::MakeFunction("castDECIMALNullOnOverflow",
                                                    {n_add_0}, decimal128(13, 2));
  auto n_mul_1 = TreeExprBuilder::MakeFunction("multiply", {n_mul_0, n_cast_dec_1},
                                               decimal128(38, 6));
  ///
  auto n_project_func = TreeExprBuilder::MakeFunction(
      "codegen_project",
      {gandiva_field_list[4], gandiva_field_list[5], gandiva_field_list[0],
       gandiva_field_list[1], n_mul_0, n_mul_1, gandiva_field_list[0],
       gandiva_field_list[1], gandiva_field_list[2]},
      uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  // Filter
  auto n_filter_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_date = TreeExprBuilder::MakeFunction(
      "castDATE", {TreeExprBuilder::MakeLiteral((int32_t)10471)}, arrow::date32());
  auto n_filter_func = TreeExprBuilder::MakeFunction(
      "less_than_or_equal_to", {gandiva_field_list[6], n_date}, arrow::boolean());
  auto n_filter =
      TreeExprBuilder::MakeFunction("filter", {n_filter_input, n_filter_func}, uint32());
  // CondProject
  auto n_cond_project =
      TreeExprBuilder::MakeFunction("CondProject", {n_project, n_filter}, uint32());
  auto n_cond_project_child =
      TreeExprBuilder::MakeFunction("child", {n_cond_project}, uint32());

  // Aggregate
  auto n_groupby_0 =
      TreeExprBuilder::MakeFunction("action_groupby", {gandiva_field_list[4]}, uint32());
  auto n_groupby_1 =
      TreeExprBuilder::MakeFunction("action_groupby", {gandiva_field_list[5]}, uint32());
  auto n_sum_0 = TreeExprBuilder::MakeFunction("action_sum_partial",
                                               {gandiva_field_list[0]}, uint32());
  auto n_sum_1 = TreeExprBuilder::MakeFunction("action_sum_partial",
                                               {gandiva_field_list[1]}, uint32());
  /// agg action
  auto f_mul_0 = field("mul_0", decimal128(26, 4));
  auto f_mul_1 = field("mul_1", decimal128(38, 6));
  auto n_sum_2 = TreeExprBuilder::MakeFunction(
      "action_sum_partial", {TreeExprBuilder::MakeField(f_mul_0)}, uint32());
  auto n_sum_3 = TreeExprBuilder::MakeFunction(
      "action_sum_partial", {TreeExprBuilder::MakeField(f_mul_1)}, uint32());
  ///
  auto n_sum_count_0 = TreeExprBuilder::MakeFunction("action_sum_count",
                                                     {gandiva_field_list[0]}, uint32());
  auto n_sum_count_1 = TreeExprBuilder::MakeFunction("action_sum_count",
                                                     {gandiva_field_list[1]}, uint32());
  auto n_sum_count_2 = TreeExprBuilder::MakeFunction("action_sum_count",
                                                     {gandiva_field_list[2]}, uint32());
  auto n_count_0 = TreeExprBuilder::MakeFunction("action_countLiteral_1", {}, uint32());
  auto n_action = TreeExprBuilder::MakeFunction(
      "aggregateActions",
      {n_groupby_0, n_groupby_1, n_sum_0, n_sum_1, n_sum_2, n_sum_3, n_sum_count_0,
       n_sum_count_1, n_sum_count_2, n_count_0},
      uint32());
  // // res fields
  auto f_sum_0 = field("sum_0", decimal128(22, 2));
  auto f_isEmpty_0 = field("isEmpty_0", boolean());
  auto f_sum_1 = field("sum_1", decimal128(22, 2));
  auto f_isEmpty_1 = field("isEmpty_1", boolean());
  auto f_sum_2 = field("sum_2", decimal128(36, 4));
  auto f_isEmpty_2 = field("isEmpty_2", boolean());
  auto f_sum_3 = field("sum_3", decimal128(38, 6));
  auto f_isEmpty_3 = field("isEmpty_3", boolean());
  auto f_sum_4 = field("sum_4", decimal128(22, 2));
  auto f_count_0 = field("count_0", int64());
  auto f_sum_5 = field("sum_5", decimal128(22, 2));
  auto f_count_1 = field("count_1", int64());
  auto f_sum_6 = field("sum_6", decimal128(22, 2));
  auto f_count_2 = field("count_2", int64());
  auto f_count_3 = field("count_3", int64());

  // aggregate expressions
  auto n_proj = TreeExprBuilder::MakeFunction(
      "aggregateExpressions",
      {gandiva_field_list[4], gandiva_field_list[5], gandiva_field_list[0],
       gandiva_field_list[1], TreeExprBuilder::MakeField(f_mul_0),
       TreeExprBuilder::MakeField(f_mul_1), gandiva_field_list[0], gandiva_field_list[1],
       gandiva_field_list[2]},
      uint32());

  std::vector<std::shared_ptr<::gandiva::Node>> agg_res_fields_list = {
      gandiva_field_list[4],
      gandiva_field_list[5],
      TreeExprBuilder::MakeField(f_sum_0),
      TreeExprBuilder::MakeField(f_isEmpty_0),
      TreeExprBuilder::MakeField(f_sum_1),
      TreeExprBuilder::MakeField(f_isEmpty_1),
      TreeExprBuilder::MakeField(f_sum_2),
      TreeExprBuilder::MakeField(f_isEmpty_2),
      TreeExprBuilder::MakeField(f_sum_3),
      TreeExprBuilder::MakeField(f_isEmpty_3),
      TreeExprBuilder::MakeField(f_sum_4),
      TreeExprBuilder::MakeField(f_count_0),
      TreeExprBuilder::MakeField(f_sum_5),
      TreeExprBuilder::MakeField(f_count_1),
      TreeExprBuilder::MakeField(f_sum_6),
      TreeExprBuilder::MakeField(f_count_2),
      TreeExprBuilder::MakeField(f_count_3)};
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

  auto f0_name = field_list_[0][0]->name();
  auto f1_name = field_list_[0][1]->name();
  auto f2_name = field_list_[0][2]->name();
  auto f3_name = field_list_[0][3]->name();
  auto f4_name = field_list_[0][4]->name();
  auto f5_name = field_list_[0][5]->name();

  auto f0_type = field_list_[0][0]->type();
  auto f1_type = field_list_[0][1]->type();
  auto f2_type = field_list_[0][2]->type();
  auto f3_type = field_list_[0][3]->type();
  auto f4_type = field_list_[0][4]->type();
  auto f5_type = field_list_[0][5]->type();

  ret_field_list_ = {field(f4_name, f4_type),
                     field(f5_name, f5_type),
                     f_sum_0,
                     f_isEmpty_0,
                     f_sum_1,
                     f_isEmpty_1,
                     f_sum_2,
                     f_isEmpty_2,
                     f_sum_3,
                     f_isEmpty_3,
                     f_sum_4,
                     f_count_0,
                     f_sum_5,
                     f_count_1,
                     f_sum_6,
                     f_count_2,
                     f_count_3};

  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

}  // namespace codegen
}  // namespace sparkcolumnarplugin
