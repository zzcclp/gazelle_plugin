// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: selection.proto

package io.substrait;

public interface FieldReferenceOrBuilder extends
    // @@protoc_insertion_point(interface_extends:io.substrait.FieldReference)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>.io.substrait.ReferenceSegment direct_reference = 1;</code>
   * @return Whether the directReference field is set.
   */
  boolean hasDirectReference();
  /**
   * <code>.io.substrait.ReferenceSegment direct_reference = 1;</code>
   * @return The directReference.
   */
  io.substrait.ReferenceSegment getDirectReference();
  /**
   * <code>.io.substrait.ReferenceSegment direct_reference = 1;</code>
   */
  io.substrait.ReferenceSegmentOrBuilder getDirectReferenceOrBuilder();

  /**
   * <code>.io.substrait.MaskExpression masked_reference = 2;</code>
   * @return Whether the maskedReference field is set.
   */
  boolean hasMaskedReference();
  /**
   * <code>.io.substrait.MaskExpression masked_reference = 2;</code>
   * @return The maskedReference.
   */
  io.substrait.MaskExpression getMaskedReference();
  /**
   * <code>.io.substrait.MaskExpression masked_reference = 2;</code>
   */
  io.substrait.MaskExpressionOrBuilder getMaskedReferenceOrBuilder();

  public io.substrait.FieldReference.ReferenceTypeCase getReferenceTypeCase();
}