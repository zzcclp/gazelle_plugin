// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: relations.proto

package io.substrait;

public interface ReadRelOrBuilder extends
    // @@protoc_insertion_point(interface_extends:io.substrait.ReadRel)
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
   * <code>.io.substrait.Type.NamedStruct base_schema = 2;</code>
   * @return Whether the baseSchema field is set.
   */
  boolean hasBaseSchema();
  /**
   * <code>.io.substrait.Type.NamedStruct base_schema = 2;</code>
   * @return The baseSchema.
   */
  io.substrait.Type.NamedStruct getBaseSchema();
  /**
   * <code>.io.substrait.Type.NamedStruct base_schema = 2;</code>
   */
  io.substrait.Type.NamedStructOrBuilder getBaseSchemaOrBuilder();

  /**
   * <code>.io.substrait.Expression filter = 3;</code>
   * @return Whether the filter field is set.
   */
  boolean hasFilter();
  /**
   * <code>.io.substrait.Expression filter = 3;</code>
   * @return The filter.
   */
  io.substrait.Expression getFilter();
  /**
   * <code>.io.substrait.Expression filter = 3;</code>
   */
  io.substrait.ExpressionOrBuilder getFilterOrBuilder();

  /**
   * <code>.io.substrait.MaskExpression projection = 4;</code>
   * @return Whether the projection field is set.
   */
  boolean hasProjection();
  /**
   * <code>.io.substrait.MaskExpression projection = 4;</code>
   * @return The projection.
   */
  io.substrait.MaskExpression getProjection();
  /**
   * <code>.io.substrait.MaskExpression projection = 4;</code>
   */
  io.substrait.MaskExpressionOrBuilder getProjectionOrBuilder();

  /**
   * <code>.io.substrait.ReadRel.VirtualTable virtual_table = 5;</code>
   * @return Whether the virtualTable field is set.
   */
  boolean hasVirtualTable();
  /**
   * <code>.io.substrait.ReadRel.VirtualTable virtual_table = 5;</code>
   * @return The virtualTable.
   */
  io.substrait.ReadRel.VirtualTable getVirtualTable();
  /**
   * <code>.io.substrait.ReadRel.VirtualTable virtual_table = 5;</code>
   */
  io.substrait.ReadRel.VirtualTableOrBuilder getVirtualTableOrBuilder();

  /**
   * <code>.io.substrait.ReadRel.LocalFiles local_files = 6;</code>
   * @return Whether the localFiles field is set.
   */
  boolean hasLocalFiles();
  /**
   * <code>.io.substrait.ReadRel.LocalFiles local_files = 6;</code>
   * @return The localFiles.
   */
  io.substrait.ReadRel.LocalFiles getLocalFiles();
  /**
   * <code>.io.substrait.ReadRel.LocalFiles local_files = 6;</code>
   */
  io.substrait.ReadRel.LocalFilesOrBuilder getLocalFilesOrBuilder();

  /**
   * <code>.io.substrait.ReadRel.NamedTable named_table = 7;</code>
   * @return Whether the namedTable field is set.
   */
  boolean hasNamedTable();
  /**
   * <code>.io.substrait.ReadRel.NamedTable named_table = 7;</code>
   * @return The namedTable.
   */
  io.substrait.ReadRel.NamedTable getNamedTable();
  /**
   * <code>.io.substrait.ReadRel.NamedTable named_table = 7;</code>
   */
  io.substrait.ReadRel.NamedTableOrBuilder getNamedTableOrBuilder();

  public io.substrait.ReadRel.ReadTypeCase getReadTypeCase();
}
