// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: selection.proto

package io.substrait;

public interface ReferenceSegmentOrBuilder extends
    // @@protoc_insertion_point(interface_extends:io.substrait.ReferenceSegment)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>.io.substrait.ReferenceSegment.MapKey map_key = 1;</code>
   * @return Whether the mapKey field is set.
   */
  boolean hasMapKey();
  /**
   * <code>.io.substrait.ReferenceSegment.MapKey map_key = 1;</code>
   * @return The mapKey.
   */
  io.substrait.ReferenceSegment.MapKey getMapKey();
  /**
   * <code>.io.substrait.ReferenceSegment.MapKey map_key = 1;</code>
   */
  io.substrait.ReferenceSegment.MapKeyOrBuilder getMapKeyOrBuilder();

  /**
   * <code>.io.substrait.ReferenceSegment.MapKeyExpression expression = 2;</code>
   * @return Whether the expression field is set.
   */
  boolean hasExpression();
  /**
   * <code>.io.substrait.ReferenceSegment.MapKeyExpression expression = 2;</code>
   * @return The expression.
   */
  io.substrait.ReferenceSegment.MapKeyExpression getExpression();
  /**
   * <code>.io.substrait.ReferenceSegment.MapKeyExpression expression = 2;</code>
   */
  io.substrait.ReferenceSegment.MapKeyExpressionOrBuilder getExpressionOrBuilder();

  /**
   * <code>.io.substrait.ReferenceSegment.StructField struct_field = 3;</code>
   * @return Whether the structField field is set.
   */
  boolean hasStructField();
  /**
   * <code>.io.substrait.ReferenceSegment.StructField struct_field = 3;</code>
   * @return The structField.
   */
  io.substrait.ReferenceSegment.StructField getStructField();
  /**
   * <code>.io.substrait.ReferenceSegment.StructField struct_field = 3;</code>
   */
  io.substrait.ReferenceSegment.StructFieldOrBuilder getStructFieldOrBuilder();

  /**
   * <code>.io.substrait.ReferenceSegment.ListElement list_element = 4;</code>
   * @return Whether the listElement field is set.
   */
  boolean hasListElement();
  /**
   * <code>.io.substrait.ReferenceSegment.ListElement list_element = 4;</code>
   * @return The listElement.
   */
  io.substrait.ReferenceSegment.ListElement getListElement();
  /**
   * <code>.io.substrait.ReferenceSegment.ListElement list_element = 4;</code>
   */
  io.substrait.ReferenceSegment.ListElementOrBuilder getListElementOrBuilder();

  /**
   * <code>.io.substrait.ReferenceSegment.ListRange list_range = 5;</code>
   * @return Whether the listRange field is set.
   */
  boolean hasListRange();
  /**
   * <code>.io.substrait.ReferenceSegment.ListRange list_range = 5;</code>
   * @return The listRange.
   */
  io.substrait.ReferenceSegment.ListRange getListRange();
  /**
   * <code>.io.substrait.ReferenceSegment.ListRange list_range = 5;</code>
   */
  io.substrait.ReferenceSegment.ListRangeOrBuilder getListRangeOrBuilder();

  public io.substrait.ReferenceSegment.ReferenceTypeCase getReferenceTypeCase();
}
