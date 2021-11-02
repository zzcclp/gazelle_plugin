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
#include <arrow/util/range.h>
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

namespace sparkcolumnarplugin {
namespace codegen {

class WholeStageTransformTest : public ::testing::Test {
 public:
  void SetUp() override {
    SetUpDataSource("tpcds_sf1000_webreturns.parquet");
    SetUpDataSource("tpcds_sf1000_websales.parquet");
  }

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
      ASSERT_NOT_OK(parquet_reader_->GetRecordBatchReader(all_row_groups, {0, 1, 2},
                                                          &record_batch_reader_));

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

    std::vector<std::shared_ptr<::arrow::Field>> getFieldList() { return field_list_; }

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

TEST_F(WholeStageTransformTest, AggregateTest) {
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> gandiva_field_list;
  for (auto field : field_list_[0]) {
    gandiva_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  auto n_max =
      TreeExprBuilder::MakeFunction("action_max", {gandiva_field_list[0]}, uint32());

  auto n_proj =
      TreeExprBuilder::MakeFunction("aggregateExpressions", gandiva_field_list, uint32());
  auto n_action = TreeExprBuilder::MakeFunction("aggregateActions", {n_max}, uint32());
  auto n_result =
      TreeExprBuilder::MakeFunction("resultSchema", {gandiva_field_list[0]}, uint32());
  auto n_result_expr = TreeExprBuilder::MakeFunction("resultExpressions",
                                                     {gandiva_field_list[0]}, uint32());
  auto n_aggr = TreeExprBuilder::MakeFunction(
      "hashAggregateArrays", {n_proj, n_action, n_result, n_result_expr}, uint32());
  auto n_child = TreeExprBuilder::MakeFunction("child", {n_aggr}, uint32());
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());

  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);

  std::shared_ptr<arrow::Schema> schema;
  schema = arrow::schema(field_list_[0]);

  auto f0_name = field_list_[0][0]->name();
  auto f1_name = field_list_[0][1]->name();
  auto f2_name = field_list_[0][2]->name();
  auto f0_type = field_list_[0][0]->type();
  // ret_field_list_ = {field(f0_name, f0_type), field(f1_name + "_sum", int64()),
  //                   field(f2_name + "_sum", int64())};
  ret_field_list_ = {field(f0_name, f0_type)};
  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

TEST_F(WholeStageTransformTest, SHJTest) {
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> table_0_field_list;
  std::vector<std::shared_ptr<::gandiva::Node>> table_1_field_list;
  for (auto field : field_list_[0]) {
    table_0_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  for (auto field : field_list_[1]) {
    table_1_field_list.push_back(TreeExprBuilder::MakeField(field));
  }

  ///////////////////////////////////////////
  auto n_left =
      TreeExprBuilder::MakeFunction("codegen_left_schema", table_0_field_list, uint32());
  auto n_right =
      TreeExprBuilder::MakeFunction("codegen_right_schema", table_1_field_list, uint32());
  auto n_left_key = TreeExprBuilder::MakeFunction("codegen_left_key_schema",
                                                  {table_0_field_list[0]}, uint32());
  auto n_right_key = TreeExprBuilder::MakeFunction("codegen_right_key_schema",
                                                   {table_1_field_list[0]}, uint32());
  auto n_result = TreeExprBuilder::MakeFunction(
      "result",
      {table_0_field_list[0], table_0_field_list[1], table_0_field_list[2],
       table_1_field_list[0], table_1_field_list[1], table_1_field_list[2]},
      uint32());
  auto n_hash_config = TreeExprBuilder::MakeFunction(
      "build_keys_config_node", {TreeExprBuilder::MakeLiteral((int)1)}, uint32());
  auto n_probeArrays = TreeExprBuilder::MakeFunction(
      "conditionedProbeArraysInner",
      {n_left, n_right, n_left_key, n_right_key, n_result, n_hash_config}, uint32());
  auto n_child = TreeExprBuilder::MakeFunction("child", {n_probeArrays}, uint32());
  auto n_ws = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());
  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_ws, f_res);

  auto schema_table_0 = arrow::schema(field_list_[0]);
  auto schema_table_1 = arrow::schema(field_list_[1]);
  auto schema_res =
      arrow::schema({field_list_[0][0], field_list_[0][1], field_list_[0][2],
                     field_list_[1][0], field_list_[1][1], field_list_[1][2]});
  ////////////////////// Build side ///////////////////////
  auto n_hash_kernel = TreeExprBuilder::MakeFunction(
      "HashRelation", {n_left_key, n_hash_config}, uint32());
  auto n_hash = TreeExprBuilder::MakeFunction("standalone", {n_hash_kernel}, uint32());
  auto hashRelation_expr = TreeExprBuilder::MakeExpression(n_hash, f_res);
  std::shared_ptr<CodeGenerator> expr_build;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), schema_table_0,
                                    {hashRelation_expr}, {}, &expr_build, true));
  ////////////////////// Probe side ///////////////////////
  std::shared_ptr<CodeGenerator> ws_generator;
  ASSERT_NOT_OK(
      CreateCodeGenerator(ctx.memory_pool(), schema_table_1, {ws_expr},
                          {field_list_[0][0], field_list_[0][1], field_list_[0][2],
                           field_list_[1][0], field_list_[1][1], field_list_[1][2]},
                          &ws_generator, true));
  ///////////////////// Calculation //////////////////
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  auto build_iter = ds_iter_list_[0];
  auto typed_build_iter =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(build_iter);
  while (typed_build_iter->HasNext()) {
    std::shared_ptr<arrow::RecordBatch> batch;
    typed_build_iter->Next(&batch);
    ASSERT_NOT_OK(expr_build->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIteratorBase> build_result_iterator;
  ASSERT_NOT_OK(expr_build->finish(&build_result_iterator));

  StartWithIterator(ws_generator, {ds_iter_list_[1], build_result_iterator});
}

TEST_F(WholeStageTransformTest, AggregateAndSHJTest) {
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> table_0_field_list;
  std::vector<std::shared_ptr<::gandiva::Node>> table_1_field_list;
  for (auto field : field_list_[0]) {
    table_0_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  for (auto field : field_list_[1]) {
    table_1_field_list.push_back(TreeExprBuilder::MakeField(field));
  }

  ////////////////////// SHJ ///////////////////////
  auto n_left = TreeExprBuilder::MakeFunction(
      "codegen_left_schema",
      {table_0_field_list[0], table_0_field_list[1], table_0_field_list[2]}, uint32());
  auto n_right = TreeExprBuilder::MakeFunction(
      "codegen_right_schema",
      {table_1_field_list[0], table_1_field_list[1], table_1_field_list[2]}, uint32());
  auto n_left_key = TreeExprBuilder::MakeFunction("codegen_left_key_schema",
                                                  {table_0_field_list[0]}, uint32());
  auto n_right_key = TreeExprBuilder::MakeFunction("codegen_right_key_schema",
                                                   {table_1_field_list[0]}, uint32());
  auto n_join_result = TreeExprBuilder::MakeFunction(
      "result",
      {table_0_field_list[0], table_0_field_list[1], table_0_field_list[2],
       table_1_field_list[0], table_1_field_list[1], table_1_field_list[2]},
      uint32());
  auto n_hash_config = TreeExprBuilder::MakeFunction(
      "build_keys_config_node", {TreeExprBuilder::MakeLiteral((int)1)}, uint32());
  auto n_probeArrays = TreeExprBuilder::MakeFunction(
      "conditionedProbeArraysInner",
      {n_left, n_right, n_left_key, n_right_key, n_join_result, n_hash_config}, uint32());
  auto n_shj_child = TreeExprBuilder::MakeFunction("child", {n_probeArrays}, uint32());
  ////////////////////// Aggregate ///////////////////////
  auto n_max =
      TreeExprBuilder::MakeFunction("action_max", {table_0_field_list[0]}, uint32());
  auto n_proj = TreeExprBuilder::MakeFunction(
      "aggregateExpressions",
      {table_0_field_list[0], table_0_field_list[1], table_0_field_list[2],
       table_1_field_list[0], table_1_field_list[1], table_1_field_list[2]},
      uint32());
  auto n_action = TreeExprBuilder::MakeFunction("aggregateActions", {n_max}, uint32());
  auto n_agg_result =
      TreeExprBuilder::MakeFunction("resultSchema", {table_0_field_list[0]}, uint32());
  auto n_result_expr = TreeExprBuilder::MakeFunction("resultExpressions",
                                                     {table_0_field_list[0]}, uint32());
  auto n_aggr = TreeExprBuilder::MakeFunction(
      "hashAggregateArrays", {n_proj, n_action, n_agg_result, n_result_expr}, uint32());

  auto n_child = TreeExprBuilder::MakeFunction("child", {n_aggr, n_shj_child}, uint32());
  auto n_ws = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());
  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_ws, f_res);

  auto schema_table_0 = arrow::schema(field_list_[0]);
  auto schema_table_1 = arrow::schema(field_list_[1]);
  auto schema_res =
      arrow::schema({field_list_[0][0], field_list_[0][1], field_list_[0][2],
                     field_list_[1][0], field_list_[1][1], field_list_[1][2]});
  ////////////////////// SHJ Build side ///////////////////////
  auto n_hash_kernel = TreeExprBuilder::MakeFunction(
      "HashRelation", {n_left_key, n_hash_config}, uint32());
  auto n_hash = TreeExprBuilder::MakeFunction("standalone", {n_hash_kernel}, uint32());
  auto hashRelation_expr = TreeExprBuilder::MakeExpression(n_hash, f_res);
  std::shared_ptr<CodeGenerator> expr_build;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), schema_table_0,
                                    {hashRelation_expr}, {}, &expr_build, true));
  ////////////////////// WS Transform ///////////////////////
  std::shared_ptr<CodeGenerator> ws_generator;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), schema_table_1, {ws_expr},
                                    {field_list_[0][0]}, &ws_generator, true));
  ///////////////////// Calculation //////////////////
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  auto build_iter = ds_iter_list_[0];
  auto typed_build_iter =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(build_iter);
  while (typed_build_iter->HasNext()) {
    std::shared_ptr<arrow::RecordBatch> batch;
    typed_build_iter->Next(&batch);
    ASSERT_NOT_OK(expr_build->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIteratorBase> build_result_iterator;
  ASSERT_NOT_OK(expr_build->finish(&build_result_iterator));

  StartWithIterator(ws_generator, {ds_iter_list_[1], build_result_iterator});
}

TEST_F(WholeStageTransformTest, ProjectTest) {
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> gandiva_field_list;
  for (auto field : field_list_[0]) {
    gandiva_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  auto n_project_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_project_func = TreeExprBuilder::MakeFunction(
      "codegen_project", {gandiva_field_list[0], gandiva_field_list[1]}, uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  auto n_child = TreeExprBuilder::MakeFunction("child", {n_project}, uint32());
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());

  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);

  std::shared_ptr<arrow::Schema> schema;
  schema = arrow::schema(field_list_[0]);

  auto f0_name = field_list_[0][0]->name();
  auto f1_name = field_list_[0][1]->name();
  auto f0_type = field_list_[0][0]->type();
  auto f1_type = field_list_[0][1]->type();

  ret_field_list_ = {field(f0_name, f0_type), field(f1_name, f1_type)};
  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

TEST_F(WholeStageTransformTest, FilterTest) {
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> gandiva_field_list;
  for (auto field : field_list_[0]) {
    gandiva_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  auto n_filter_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_filter_func = TreeExprBuilder::MakeFunction(
      "less_than_or_equal_to",
      {gandiva_field_list[1], TreeExprBuilder::MakeLiteral((int32_t)50000)},
      arrow::boolean());
  auto n_filter =
      TreeExprBuilder::MakeFunction("filter", {n_filter_input, n_filter_func}, uint32());
  auto n_child = TreeExprBuilder::MakeFunction("child", {n_filter}, uint32());
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());

  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);

  std::shared_ptr<arrow::Schema> schema;
  schema = arrow::schema(field_list_[0]);

  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, field_list_[0], &ws_generator,
                      true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

TEST_F(WholeStageTransformTest, CondProjectTest) {
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
      "codegen_project", {gandiva_field_list[0], gandiva_field_list[1]}, uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  // Filter
  auto n_filter_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_filter_func = TreeExprBuilder::MakeFunction(
      "less_than_or_equal_to",
      {gandiva_field_list[1], TreeExprBuilder::MakeLiteral((int32_t)50000)},
      arrow::boolean());
  auto n_filter =
      TreeExprBuilder::MakeFunction("filter", {n_filter_input, n_filter_func}, uint32());
  // CondProject
  auto n_cond_project =
      TreeExprBuilder::MakeFunction("CondProject", {n_project, n_filter}, uint32());

  auto n_child = TreeExprBuilder::MakeFunction("child", {n_cond_project}, uint32());
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());

  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);

  std::shared_ptr<arrow::Schema> schema;
  schema = arrow::schema(field_list_[0]);

  auto f0_name = field_list_[0][0]->name();
  auto f1_name = field_list_[0][1]->name();
  auto f0_type = field_list_[0][0]->type();
  auto f1_type = field_list_[0][1]->type();

  ret_field_list_ = {field(f0_name, f0_type), field(f1_name, f1_type)};
  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

TEST_F(WholeStageTransformTest, CondProjectAggTest) {
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
      "codegen_project", {gandiva_field_list[0], gandiva_field_list[1]}, uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  // Filter
  auto n_filter_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_filter_func = TreeExprBuilder::MakeFunction(
      "less_than_or_equal_to",
      {gandiva_field_list[1], TreeExprBuilder::MakeLiteral((int32_t)50000)},
      arrow::boolean());
  auto n_filter =
      TreeExprBuilder::MakeFunction("filter", {n_filter_input, n_filter_func}, uint32());
  // CondProject
  auto n_cond_project =
      TreeExprBuilder::MakeFunction("CondProject", {n_project, n_filter}, uint32());
  auto n_cond_project_child =
      TreeExprBuilder::MakeFunction("child", {n_cond_project}, uint32());
  // Agg
  auto n_max_0 =
      TreeExprBuilder::MakeFunction("action_max", {gandiva_field_list[0]}, uint32());
  auto n_max_1 =
      TreeExprBuilder::MakeFunction("action_max", {gandiva_field_list[1]}, uint32());
  auto n_action =
      TreeExprBuilder::MakeFunction("aggregateActions", {n_max_0, n_max_1}, uint32());

  auto f_max_0 = field("max_0", int32());
  auto f_max_1 = field("max_1", int32());

  auto n_proj = TreeExprBuilder::MakeFunction(
      "aggregateExpressions", {gandiva_field_list[0], gandiva_field_list[1]}, uint32());

  std::vector<std::shared_ptr<::gandiva::Node>> agg_res_fields_list = {
      TreeExprBuilder::MakeField(f_max_0), TreeExprBuilder::MakeField(f_max_1)};
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

  std::shared_ptr<arrow::Schema> schema = arrow::schema(field_list_[0]);

  ret_field_list_ = {f_max_0, f_max_1};
  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

TEST_F(WholeStageTransformTest, CondProjAggAndSHJTest) {
  /* Structure of TreeNode in this test case:
                        wholestagetransform
                               |
                             child
                          /         \
                      project      child
                                  /      \
                              aggregate  child
                                         /    \
                                      filter  child
                                                |
                                               join
  */
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> table_0_field_list;
  std::vector<std::shared_ptr<::gandiva::Node>> table_1_field_list;
  for (auto field : field_list_[0]) {
    table_0_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  for (auto field : field_list_[1]) {
    table_1_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  auto schema_table_0 = arrow::schema(field_list_[0]);
  auto schema_table_1 = arrow::schema(field_list_[1]);
  auto schema_res =
      arrow::schema({field_list_[0][0], field_list_[0][1], field_list_[0][2],
                     field_list_[1][0], field_list_[1][1], field_list_[1][2]});

  ////////////////////// SHJ ///////////////////////
  auto n_left = TreeExprBuilder::MakeFunction(
      "codegen_left_schema",
      {table_0_field_list[0], table_0_field_list[1], table_0_field_list[2]}, uint32());
  auto n_right = TreeExprBuilder::MakeFunction(
      "codegen_right_schema",
      {table_1_field_list[0], table_1_field_list[1], table_1_field_list[2]}, uint32());
  auto n_left_key = TreeExprBuilder::MakeFunction("codegen_left_key_schema",
                                                  {table_0_field_list[0]}, uint32());
  auto n_right_key = TreeExprBuilder::MakeFunction("codegen_right_key_schema",
                                                   {table_1_field_list[0]}, uint32());
  auto n_join_result = TreeExprBuilder::MakeFunction(
      "result",
      {table_0_field_list[0], table_0_field_list[1], table_0_field_list[2],
       table_1_field_list[0], table_1_field_list[1], table_1_field_list[2]},
      uint32());
  auto n_hash_config = TreeExprBuilder::MakeFunction(
      "build_keys_config_node", {TreeExprBuilder::MakeLiteral((int)1)}, uint32());
  auto n_probeArrays = TreeExprBuilder::MakeFunction(
      "conditionedProbeArraysInner",
      {n_left, n_right, n_left_key, n_right_key, n_join_result, n_hash_config}, uint32());
  auto n_shj_child = TreeExprBuilder::MakeFunction("child", {n_probeArrays}, uint32());
  ////////////////////// Filter ///////////////////////
  auto n_filter_input = TreeExprBuilder::MakeFunction(
      "codegen_input_schema",
      {table_0_field_list[0], table_0_field_list[1], table_0_field_list[2],
       table_1_field_list[0], table_1_field_list[1], table_1_field_list[2]},
      uint32());
  auto n_filter_func = TreeExprBuilder::MakeFunction(
      "less_than_or_equal_to",
      {table_0_field_list[1], TreeExprBuilder::MakeLiteral((int32_t)50000)},
      arrow::boolean());
  auto n_filter =
      TreeExprBuilder::MakeFunction("filter", {n_filter_input, n_filter_func}, uint32());
  auto n_filter_child =
      TreeExprBuilder::MakeFunction("child", {n_filter, n_shj_child}, uint32());
  ////////////////////// Aggregate ///////////////////////
  auto n_max =
      TreeExprBuilder::MakeFunction("action_max", {table_0_field_list[1]}, uint32());
  auto n_proj = TreeExprBuilder::MakeFunction(
      "aggregateExpressions",
      {table_0_field_list[0], table_0_field_list[1], table_0_field_list[2],
       table_1_field_list[0], table_1_field_list[1], table_1_field_list[2]},
      uint32());
  auto n_action = TreeExprBuilder::MakeFunction("aggregateActions", {n_max}, uint32());
  auto n_agg_result =
      TreeExprBuilder::MakeFunction("resultSchema", {table_0_field_list[1]}, uint32());
  auto n_result_expr = TreeExprBuilder::MakeFunction("resultExpressions",
                                                     {table_0_field_list[1]}, uint32());
  auto n_aggr = TreeExprBuilder::MakeFunction(
      "hashAggregateArrays", {n_proj, n_action, n_agg_result, n_result_expr}, uint32());

  auto n_agg_child =
      TreeExprBuilder::MakeFunction("child", {n_aggr, n_filter_child}, uint32());
  ////////////////////// SHJ Build side ///////////////////////
  auto n_hash_kernel = TreeExprBuilder::MakeFunction(
      "HashRelation", {n_left_key, n_hash_config}, uint32());
  auto n_hash = TreeExprBuilder::MakeFunction("standalone", {n_hash_kernel}, uint32());
  auto hashRelation_expr = TreeExprBuilder::MakeExpression(n_hash, f_res);
  std::shared_ptr<CodeGenerator> expr_build;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), schema_table_0,
                                    {hashRelation_expr}, {}, &expr_build, true));
  ////////////////////// Project ///////////////////////
  auto n_add = TreeExprBuilder::MakeFunction(
      "add", {table_0_field_list[1], TreeExprBuilder::MakeLiteral((int32_t)20)}, int32());
  auto n_project_input = TreeExprBuilder::MakeFunction("codegen_input_schema",
                                                       {table_0_field_list[1]}, uint32());
  auto n_project_func =
      TreeExprBuilder::MakeFunction("codegen_project", {n_add}, uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  auto n_project_child =
      TreeExprBuilder::MakeFunction("child", {n_project, n_agg_child}, uint32());
  ////////////////////// WS Transform ///////////////////////
  auto n_ws =
      TreeExprBuilder::MakeFunction("wholestagetransform", {n_project_child}, uint32());
  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_ws, f_res);
  std::shared_ptr<CodeGenerator> ws_generator;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), schema_table_1, {ws_expr},
                                    {field_list_[0][1]}, &ws_generator, true));
  ///////////////////// Calculation //////////////////
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  auto build_iter = ds_iter_list_[0];
  auto typed_build_iter =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(build_iter);
  while (typed_build_iter->HasNext()) {
    std::shared_ptr<arrow::RecordBatch> batch;
    typed_build_iter->Next(&batch);
    ASSERT_NOT_OK(expr_build->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIteratorBase> build_result_iterator;
  ASSERT_NOT_OK(expr_build->finish(&build_result_iterator));

  StartWithIterator(ws_generator, {ds_iter_list_[1], build_result_iterator});
}

TEST_F(WholeStageTransformTest, SortTest) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  std::vector<std::shared_ptr<::gandiva::Node>> table_0_field_list;
  for (auto field : field_list_[0]) {
    table_0_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  auto schema_table_0 = arrow::schema(field_list_[0]);

  auto n_key_func =
      TreeExprBuilder::MakeFunction("key_function", {table_0_field_list[0]}, uint32());
  auto n_key_field =
      TreeExprBuilder::MakeFunction("key_field", {table_0_field_list[0]}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction(
      "NaN_check", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction(
      "codegen", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto n_res_fields =
      TreeExprBuilder::MakeFunction("res_fields", table_0_field_list, uint32());
  auto n_sort_to_indices =
      TreeExprBuilder::MakeFunction("sortArraysToIndices",
                                    {n_key_func, n_key_field, n_dir, n_nulls_order,
                                     NaN_check, do_codegen, n_res_fields},
                                    uint32());
  auto n_child = TreeExprBuilder::MakeFunction("child", {n_sort_to_indices}, uint32());
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());

  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);

  std::shared_ptr<arrow::Schema> schema;
  schema = arrow::schema(field_list_[0]);

  std::shared_ptr<CodeGenerator> ws_generator;
  arrow::compute::ExecContext ctx;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, field_list_[0], &ws_generator,
                      true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[0]});
}

TEST_F(WholeStageTransformTest, SimpleSMJTest) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  std::vector<std::shared_ptr<::gandiva::Node>> table_0_field_list;
  std::vector<std::shared_ptr<::gandiva::Node>> table_1_field_list;
  for (auto field : field_list_[0]) {
    table_0_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  for (auto field : field_list_[1]) {
    table_1_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  auto schema_table_0 = arrow::schema(field_list_[0]);
  auto schema_table_1 = arrow::schema(field_list_[1]);

  ////////////////////// SMJ //////////////////////
  auto n_semi_left =
      TreeExprBuilder::MakeFunction("codegen_left_schema", table_0_field_list, uint32());
  auto n_semi_right =
      TreeExprBuilder::MakeFunction("codegen_right_schema", table_1_field_list, uint32());
  auto n_semi_left_key = TreeExprBuilder::MakeFunction("codegen_left_key_schema",
                                                       {table_0_field_list[1]}, uint32());
  auto n_semi_right_key = TreeExprBuilder::MakeFunction(
      "codegen_right_key_schema", {table_1_field_list[1]}, uint32());
  auto n_semi_result = TreeExprBuilder::MakeFunction(
      "result", {table_1_field_list[0], table_1_field_list[1]}, uint32());
  auto n_semi_probeArrays = TreeExprBuilder::MakeFunction(
      "conditionedMergeJoinSemi",
      {n_semi_left, n_semi_right, n_semi_left_key, n_semi_right_key, n_semi_result},
      uint32());
  auto n_child = TreeExprBuilder::MakeFunction("child", {n_semi_probeArrays}, uint32());
  ////////////////////// Sort //////////////////////
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction(
      "NaN_check", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction(
      "codegen", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  // Sort left
  auto n_left_key_func =
      TreeExprBuilder::MakeFunction("key_function", {table_0_field_list[1]}, uint32());
  auto n_left_key_field =
      TreeExprBuilder::MakeFunction("key_field", {table_0_field_list[1]}, uint32());
  auto n_left_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_left_key_func, n_left_key_field, n_dir, n_nulls_order, NaN_check, do_codegen},
      uint32());
  auto n_sort_left =
      TreeExprBuilder::MakeFunction("standalone", {n_left_sort_to_indices}, uint32());
  auto sortArrays_expr_left = TreeExprBuilder::MakeExpression(n_sort_left, f_res);
  std::shared_ptr<CodeGenerator> expr_sort_left;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), schema_table_0,
                                    {sortArrays_expr_left}, field_list_[0],
                                    &expr_sort_left, true));
  std::shared_ptr<ResultIteratorBase> left_sort_iterator;
  ASSERT_NOT_OK(expr_sort_left->finish(&left_sort_iterator));
  // Sort right
  auto n_right_key_func =
      TreeExprBuilder::MakeFunction("key_function", {table_1_field_list[1]}, uint32());
  auto n_right_key_field =
      TreeExprBuilder::MakeFunction("key_field", {table_1_field_list[1]}, uint32());
  auto n_right_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_right_key_func, n_right_key_field, n_dir, n_nulls_order, NaN_check, do_codegen},
      uint32());
  auto n_sort_right =
      TreeExprBuilder::MakeFunction("standalone", {n_right_sort_to_indices}, uint32());
  auto sortArrays_expr_right = TreeExprBuilder::MakeExpression(n_sort_right, f_res);
  std::shared_ptr<CodeGenerator> expr_sort_right;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), schema_table_1,
                                    {sortArrays_expr_right}, field_list_[1],
                                    &expr_sort_right, true));
  std::shared_ptr<ResultIteratorBase> right_sort_iterator;
  ASSERT_NOT_OK(expr_sort_right->finish(&right_sort_iterator));
  ////////////////////////////////////////////
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());
  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);
  std::shared_ptr<CodeGenerator> ws_generator;
  CreateCodeGenerator(ctx.memory_pool(), arrow::schema({}), {ws_expr},
                      {field_list_[1][0], field_list_[1][1]}, &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {ds_iter_list_[1], ds_iter_list_[0],
                                   right_sort_iterator, left_sort_iterator});
}

TEST_F(WholeStageTransformTest, ProjectWithLazyReadTest) {
  num_batches = 0;
  ////////////////////// prepare expr_vector ///////////////////////
  f_res = field("res", arrow::uint64());

  std::vector<std::shared_ptr<::gandiva::Node>> gandiva_field_list;
  for (auto field : field_list_[0]) {
    gandiva_field_list.push_back(TreeExprBuilder::MakeField(field));
  }
  auto n_project_input =
      TreeExprBuilder::MakeFunction("codegen_input_schema", gandiva_field_list, uint32());
  auto n_project_func = TreeExprBuilder::MakeFunction(
      "codegen_project", {gandiva_field_list[0], gandiva_field_list[1]}, uint32());
  auto n_project = TreeExprBuilder::MakeFunction(
      "project", {n_project_input, n_project_func}, uint32());
  auto n_child = TreeExprBuilder::MakeFunction("child", {n_project}, uint32());
  auto n_wscg = TreeExprBuilder::MakeFunction("wholestagetransform", {n_child}, uint32());

  ::gandiva::ExpressionPtr ws_expr = TreeExprBuilder::MakeExpression(n_wscg, f_res);

  // Lazy read
  auto n_input_field =
      TreeExprBuilder::MakeFunction("input_field", gandiva_field_list, uint32());
  auto n_lazy_read = TreeExprBuilder::MakeFunction(
      "standalone",
      {TreeExprBuilder::MakeFunction("LazyRead", {n_input_field}, uint32())}, uint32());
  auto lazy_read_expr = TreeExprBuilder::MakeExpression(n_lazy_read, f_res);

  arrow::compute::ExecContext ctx;
  std::shared_ptr<arrow::Schema> schema = arrow::schema(field_list_[0]);
  std::shared_ptr<CodeGenerator> lazy_read_handler;
  CreateCodeGenerator(ctx.memory_pool(), schema, {lazy_read_expr}, field_list_[0],
                      &lazy_read_handler, true);
  auto ds_iter =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(ds_iter_list_[0]);

  arrow::RecordBatchIterator rb_iter = arrow::MakeFunctionIterator(
      [ds_iter]() -> arrow::Result<std::shared_ptr<arrow::RecordBatch>> {
        if (!ds_iter->HasNext()) {
          return nullptr;
        }
        std::shared_ptr<RecordBatch> batch;
        ds_iter->Next(&batch);
        return batch;
      });
  lazy_read_handler->evaluate(std::move(rb_iter));
  std::shared_ptr<ResultIteratorBase> lazy_read_res_iter;
  lazy_read_handler->finish(&lazy_read_res_iter);

  auto f0_name = field_list_[0][0]->name();
  auto f1_name = field_list_[0][1]->name();
  auto f0_type = field_list_[0][0]->type();
  auto f1_type = field_list_[0][1]->type();

  ret_field_list_ = {field(f0_name, f0_type), field(f1_name, f1_type)};
  std::shared_ptr<CodeGenerator> ws_generator;
  CreateCodeGenerator(ctx.memory_pool(), schema, {ws_expr}, ret_field_list_,
                      &ws_generator, true);

  ///////////////////// Calculation //////////////////
  StartWithIterator(ws_generator, {lazy_read_res_iter});
}

}  // namespace codegen
}  // namespace sparkcolumnarplugin
