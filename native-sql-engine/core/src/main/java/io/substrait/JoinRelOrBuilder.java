// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: relations.proto

package io.substrait;

public interface JoinRelOrBuilder extends
    // @@protoc_insertion_point(interface_extends:io.substrait.JoinRel)
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
   * <code>.io.substrait.Rel left = 2;</code>
   * @return Whether the left field is set.
   */
  boolean hasLeft();
  /**
   * <code>.io.substrait.Rel left = 2;</code>
   * @return The left.
   */
  io.substrait.Rel getLeft();
  /**
   * <code>.io.substrait.Rel left = 2;</code>
   */
  io.substrait.RelOrBuilder getLeftOrBuilder();

  /**
   * <code>.io.substrait.Rel right = 3;</code>
   * @return Whether the right field is set.
   */
  boolean hasRight();
  /**
   * <code>.io.substrait.Rel right = 3;</code>
   * @return The right.
   */
  io.substrait.Rel getRight();
  /**
   * <code>.io.substrait.Rel right = 3;</code>
   */
  io.substrait.RelOrBuilder getRightOrBuilder();

  /**
   * <code>.io.substrait.Expression expression = 4;</code>
   * @return Whether the expression field is set.
   */
  boolean hasExpression();
  /**
   * <code>.io.substrait.Expression expression = 4;</code>
   * @return The expression.
   */
  io.substrait.Expression getExpression();
  /**
   * <code>.io.substrait.Expression expression = 4;</code>
   */
  io.substrait.ExpressionOrBuilder getExpressionOrBuilder();

  /**
   * <code>.io.substrait.Expression post_join_filter = 5;</code>
   * @return Whether the postJoinFilter field is set.
   */
  boolean hasPostJoinFilter();
  /**
   * <code>.io.substrait.Expression post_join_filter = 5;</code>
   * @return The postJoinFilter.
   */
  io.substrait.Expression getPostJoinFilter();
  /**
   * <code>.io.substrait.Expression post_join_filter = 5;</code>
   */
  io.substrait.ExpressionOrBuilder getPostJoinFilterOrBuilder();
}