// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: relations.proto

package io.substrait;

public interface RelOrBuilder extends
    // @@protoc_insertion_point(interface_extends:io.substrait.Rel)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>.io.substrait.ReadRel read = 1;</code>
   * @return Whether the read field is set.
   */
  boolean hasRead();
  /**
   * <code>.io.substrait.ReadRel read = 1;</code>
   * @return The read.
   */
  io.substrait.ReadRel getRead();
  /**
   * <code>.io.substrait.ReadRel read = 1;</code>
   */
  io.substrait.ReadRelOrBuilder getReadOrBuilder();

  /**
   * <code>.io.substrait.FilterRel filter = 2;</code>
   * @return Whether the filter field is set.
   */
  boolean hasFilter();
  /**
   * <code>.io.substrait.FilterRel filter = 2;</code>
   * @return The filter.
   */
  io.substrait.FilterRel getFilter();
  /**
   * <code>.io.substrait.FilterRel filter = 2;</code>
   */
  io.substrait.FilterRelOrBuilder getFilterOrBuilder();

  /**
   * <code>.io.substrait.FetchRel fetch = 3;</code>
   * @return Whether the fetch field is set.
   */
  boolean hasFetch();
  /**
   * <code>.io.substrait.FetchRel fetch = 3;</code>
   * @return The fetch.
   */
  io.substrait.FetchRel getFetch();
  /**
   * <code>.io.substrait.FetchRel fetch = 3;</code>
   */
  io.substrait.FetchRelOrBuilder getFetchOrBuilder();

  /**
   * <code>.io.substrait.AggregateRel aggregate = 4;</code>
   * @return Whether the aggregate field is set.
   */
  boolean hasAggregate();
  /**
   * <code>.io.substrait.AggregateRel aggregate = 4;</code>
   * @return The aggregate.
   */
  io.substrait.AggregateRel getAggregate();
  /**
   * <code>.io.substrait.AggregateRel aggregate = 4;</code>
   */
  io.substrait.AggregateRelOrBuilder getAggregateOrBuilder();

  /**
   * <code>.io.substrait.SortRel sort = 5;</code>
   * @return Whether the sort field is set.
   */
  boolean hasSort();
  /**
   * <code>.io.substrait.SortRel sort = 5;</code>
   * @return The sort.
   */
  io.substrait.SortRel getSort();
  /**
   * <code>.io.substrait.SortRel sort = 5;</code>
   */
  io.substrait.SortRelOrBuilder getSortOrBuilder();

  /**
   * <code>.io.substrait.JoinRel join = 6;</code>
   * @return Whether the join field is set.
   */
  boolean hasJoin();
  /**
   * <code>.io.substrait.JoinRel join = 6;</code>
   * @return The join.
   */
  io.substrait.JoinRel getJoin();
  /**
   * <code>.io.substrait.JoinRel join = 6;</code>
   */
  io.substrait.JoinRelOrBuilder getJoinOrBuilder();

  /**
   * <code>.io.substrait.ProjectRel project = 7;</code>
   * @return Whether the project field is set.
   */
  boolean hasProject();
  /**
   * <code>.io.substrait.ProjectRel project = 7;</code>
   * @return The project.
   */
  io.substrait.ProjectRel getProject();
  /**
   * <code>.io.substrait.ProjectRel project = 7;</code>
   */
  io.substrait.ProjectRelOrBuilder getProjectOrBuilder();

  /**
   * <code>.io.substrait.SetRel set = 8;</code>
   * @return Whether the set field is set.
   */
  boolean hasSet();
  /**
   * <code>.io.substrait.SetRel set = 8;</code>
   * @return The set.
   */
  io.substrait.SetRel getSet();
  /**
   * <code>.io.substrait.SetRel set = 8;</code>
   */
  io.substrait.SetRelOrBuilder getSetOrBuilder();

  public io.substrait.Rel.RelTypeCase getRelTypeCase();
}