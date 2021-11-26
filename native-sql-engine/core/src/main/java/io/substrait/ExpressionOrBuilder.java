// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: expression.proto

package io.substrait;

public interface ExpressionOrBuilder extends
    // @@protoc_insertion_point(interface_extends:io.substrait.Expression)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>.io.substrait.Expression.Literal literal = 1;</code>
   * @return Whether the literal field is set.
   */
  boolean hasLiteral();
  /**
   * <code>.io.substrait.Expression.Literal literal = 1;</code>
   * @return The literal.
   */
  io.substrait.Expression.Literal getLiteral();
  /**
   * <code>.io.substrait.Expression.Literal literal = 1;</code>
   */
  io.substrait.Expression.LiteralOrBuilder getLiteralOrBuilder();

  /**
   * <code>.io.substrait.FieldReference selection = 2;</code>
   * @return Whether the selection field is set.
   */
  boolean hasSelection();
  /**
   * <code>.io.substrait.FieldReference selection = 2;</code>
   * @return The selection.
   */
  io.substrait.FieldReference getSelection();
  /**
   * <code>.io.substrait.FieldReference selection = 2;</code>
   */
  io.substrait.FieldReferenceOrBuilder getSelectionOrBuilder();

  /**
   * <code>.io.substrait.Expression.ScalarFunction scalar_function = 3;</code>
   * @return Whether the scalarFunction field is set.
   */
  boolean hasScalarFunction();
  /**
   * <code>.io.substrait.Expression.ScalarFunction scalar_function = 3;</code>
   * @return The scalarFunction.
   */
  io.substrait.Expression.ScalarFunction getScalarFunction();
  /**
   * <code>.io.substrait.Expression.ScalarFunction scalar_function = 3;</code>
   */
  io.substrait.Expression.ScalarFunctionOrBuilder getScalarFunctionOrBuilder();

  /**
   * <code>.io.substrait.Expression.WindowFunction window_function = 5;</code>
   * @return Whether the windowFunction field is set.
   */
  boolean hasWindowFunction();
  /**
   * <code>.io.substrait.Expression.WindowFunction window_function = 5;</code>
   * @return The windowFunction.
   */
  io.substrait.Expression.WindowFunction getWindowFunction();
  /**
   * <code>.io.substrait.Expression.WindowFunction window_function = 5;</code>
   */
  io.substrait.Expression.WindowFunctionOrBuilder getWindowFunctionOrBuilder();

  /**
   * <code>.io.substrait.Expression.IfThen if_then = 6;</code>
   * @return Whether the ifThen field is set.
   */
  boolean hasIfThen();
  /**
   * <code>.io.substrait.Expression.IfThen if_then = 6;</code>
   * @return The ifThen.
   */
  io.substrait.Expression.IfThen getIfThen();
  /**
   * <code>.io.substrait.Expression.IfThen if_then = 6;</code>
   */
  io.substrait.Expression.IfThenOrBuilder getIfThenOrBuilder();

  /**
   * <code>.io.substrait.Expression.SwitchExpression switch_expression = 7;</code>
   * @return Whether the switchExpression field is set.
   */
  boolean hasSwitchExpression();
  /**
   * <code>.io.substrait.Expression.SwitchExpression switch_expression = 7;</code>
   * @return The switchExpression.
   */
  io.substrait.Expression.SwitchExpression getSwitchExpression();
  /**
   * <code>.io.substrait.Expression.SwitchExpression switch_expression = 7;</code>
   */
  io.substrait.Expression.SwitchExpressionOrBuilder getSwitchExpressionOrBuilder();

  /**
   * <code>.io.substrait.Expression.SingularOrList singular_or_list = 8;</code>
   * @return Whether the singularOrList field is set.
   */
  boolean hasSingularOrList();
  /**
   * <code>.io.substrait.Expression.SingularOrList singular_or_list = 8;</code>
   * @return The singularOrList.
   */
  io.substrait.Expression.SingularOrList getSingularOrList();
  /**
   * <code>.io.substrait.Expression.SingularOrList singular_or_list = 8;</code>
   */
  io.substrait.Expression.SingularOrListOrBuilder getSingularOrListOrBuilder();

  /**
   * <code>.io.substrait.Expression.MultiOrList multi_or_list = 9;</code>
   * @return Whether the multiOrList field is set.
   */
  boolean hasMultiOrList();
  /**
   * <code>.io.substrait.Expression.MultiOrList multi_or_list = 9;</code>
   * @return The multiOrList.
   */
  io.substrait.Expression.MultiOrList getMultiOrList();
  /**
   * <code>.io.substrait.Expression.MultiOrList multi_or_list = 9;</code>
   */
  io.substrait.Expression.MultiOrListOrBuilder getMultiOrListOrBuilder();

  /**
   * <code>.io.substrait.Expression.Enum enum = 10;</code>
   * @return Whether the enum field is set.
   */
  boolean hasEnum();
  /**
   * <code>.io.substrait.Expression.Enum enum = 10;</code>
   * @return The enum.
   */
  io.substrait.Expression.Enum getEnum();
  /**
   * <code>.io.substrait.Expression.Enum enum = 10;</code>
   */
  io.substrait.Expression.EnumOrBuilder getEnumOrBuilder();

  public io.substrait.Expression.RexTypeCase getRexTypeCase();
}
