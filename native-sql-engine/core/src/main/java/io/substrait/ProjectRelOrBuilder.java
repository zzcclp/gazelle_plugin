// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: relations.proto

package io.substrait;

public interface ProjectRelOrBuilder extends
    // @@protoc_insertion_point(interface_extends:io.substrait.ProjectRel)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>.io.substrait.RelCommon common = 1;</code>
   * @return Whether the common field is set.
   */
  boolean hasCommon();
  /**
   * <code>.io.substrait.RelCommon common = 1;</code>
   * @return The common.
   */
  io.substrait.RelCommon getCommon();
  /**
   * <code>.io.substrait.RelCommon common = 1;</code>
   */
  io.substrait.RelCommonOrBuilder getCommonOrBuilder();

  /**
   * <code>.io.substrait.Rel input = 2;</code>
   * @return Whether the input field is set.
   */
  boolean hasInput();
  /**
   * <code>.io.substrait.Rel input = 2;</code>
   * @return The input.
   */
  io.substrait.Rel getInput();
  /**
   * <code>.io.substrait.Rel input = 2;</code>
   */
  io.substrait.RelOrBuilder getInputOrBuilder();

  /**
   * <code>repeated .io.substrait.Expression expressions = 3;</code>
   */
  java.util.List<io.substrait.Expression> 
      getExpressionsList();
  /**
   * <code>repeated .io.substrait.Expression expressions = 3;</code>
   */
  io.substrait.Expression getExpressions(int index);
  /**
   * <code>repeated .io.substrait.Expression expressions = 3;</code>
   */
  int getExpressionsCount();
  /**
   * <code>repeated .io.substrait.Expression expressions = 3;</code>
   */
  java.util.List<? extends io.substrait.ExpressionOrBuilder> 
      getExpressionsOrBuilderList();
  /**
   * <code>repeated .io.substrait.Expression expressions = 3;</code>
   */
  io.substrait.ExpressionOrBuilder getExpressionsOrBuilder(
      int index);

  /**
   * <code>repeated .io.substrait.Type input_types = 4;</code>
   */
  java.util.List<io.substrait.Type> 
      getInputTypesList();
  /**
   * <code>repeated .io.substrait.Type input_types = 4;</code>
   */
  io.substrait.Type getInputTypes(int index);
  /**
   * <code>repeated .io.substrait.Type input_types = 4;</code>
   */
  int getInputTypesCount();
  /**
   * <code>repeated .io.substrait.Type input_types = 4;</code>
   */
  java.util.List<? extends io.substrait.TypeOrBuilder> 
      getInputTypesOrBuilderList();
  /**
   * <code>repeated .io.substrait.Type input_types = 4;</code>
   */
  io.substrait.TypeOrBuilder getInputTypesOrBuilder(
      int index);

  /**
   * <code>repeated .io.substrait.Type output_types = 5;</code>
   */
  java.util.List<io.substrait.Type> 
      getOutputTypesList();
  /**
   * <code>repeated .io.substrait.Type output_types = 5;</code>
   */
  io.substrait.Type getOutputTypes(int index);
  /**
   * <code>repeated .io.substrait.Type output_types = 5;</code>
   */
  int getOutputTypesCount();
  /**
   * <code>repeated .io.substrait.Type output_types = 5;</code>
   */
  java.util.List<? extends io.substrait.TypeOrBuilder> 
      getOutputTypesOrBuilderList();
  /**
   * <code>repeated .io.substrait.Type output_types = 5;</code>
   */
  io.substrait.TypeOrBuilder getOutputTypesOrBuilder(
      int index);
}