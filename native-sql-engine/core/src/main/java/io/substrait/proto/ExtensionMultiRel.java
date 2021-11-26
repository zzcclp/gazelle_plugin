// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: substrait/relations.proto

package io.substrait.proto;

/**
 * <pre>
 * Stub to support extension with multiple inputs
 * </pre>
 *
 * Protobuf type {@code substrait.ExtensionMultiRel}
 */
public final class ExtensionMultiRel extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:substrait.ExtensionMultiRel)
    ExtensionMultiRelOrBuilder {
private static final long serialVersionUID = 0L;
  // Use ExtensionMultiRel.newBuilder() to construct.
  private ExtensionMultiRel(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ExtensionMultiRel() {
    inputs_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new ExtensionMultiRel();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ExtensionMultiRel(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new java.lang.NullPointerException();
    }
    int mutable_bitField0_ = 0;
    com.google.protobuf.UnknownFieldSet.Builder unknownFields =
        com.google.protobuf.UnknownFieldSet.newBuilder();
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          case 10: {
            io.substrait.proto.RelCommon.Builder subBuilder = null;
            if (common_ != null) {
              subBuilder = common_.toBuilder();
            }
            common_ = input.readMessage(io.substrait.proto.RelCommon.parser(), extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(common_);
              common_ = subBuilder.buildPartial();
            }

            break;
          }
          case 18: {
            if (!((mutable_bitField0_ & 0x00000001) != 0)) {
              inputs_ = new java.util.ArrayList<io.substrait.proto.Rel>();
              mutable_bitField0_ |= 0x00000001;
            }
            inputs_.add(
                input.readMessage(io.substrait.proto.Rel.parser(), extensionRegistry));
            break;
          }
          case 26: {
            com.google.protobuf.Any.Builder subBuilder = null;
            if (detail_ != null) {
              subBuilder = detail_.toBuilder();
            }
            detail_ = input.readMessage(com.google.protobuf.Any.parser(), extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(detail_);
              detail_ = subBuilder.buildPartial();
            }

            break;
          }
          default: {
            if (!parseUnknownField(
                input, unknownFields, extensionRegistry, tag)) {
              done = true;
            }
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      if (((mutable_bitField0_ & 0x00000001) != 0)) {
        inputs_ = java.util.Collections.unmodifiableList(inputs_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return io.substrait.proto.Relations.internal_static_substrait_ExtensionMultiRel_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return io.substrait.proto.Relations.internal_static_substrait_ExtensionMultiRel_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            io.substrait.proto.ExtensionMultiRel.class, io.substrait.proto.ExtensionMultiRel.Builder.class);
  }

  public static final int COMMON_FIELD_NUMBER = 1;
  private io.substrait.proto.RelCommon common_;
  /**
   * <code>.substrait.RelCommon common = 1;</code>
   * @return Whether the common field is set.
   */
  @java.lang.Override
  public boolean hasCommon() {
    return common_ != null;
  }
  /**
   * <code>.substrait.RelCommon common = 1;</code>
   * @return The common.
   */
  @java.lang.Override
  public io.substrait.proto.RelCommon getCommon() {
    return common_ == null ? io.substrait.proto.RelCommon.getDefaultInstance() : common_;
  }
  /**
   * <code>.substrait.RelCommon common = 1;</code>
   */
  @java.lang.Override
  public io.substrait.proto.RelCommonOrBuilder getCommonOrBuilder() {
    return getCommon();
  }

  public static final int INPUTS_FIELD_NUMBER = 2;
  private java.util.List<io.substrait.proto.Rel> inputs_;
  /**
   * <code>repeated .substrait.Rel inputs = 2;</code>
   */
  @java.lang.Override
  public java.util.List<io.substrait.proto.Rel> getInputsList() {
    return inputs_;
  }
  /**
   * <code>repeated .substrait.Rel inputs = 2;</code>
   */
  @java.lang.Override
  public java.util.List<? extends io.substrait.proto.RelOrBuilder> 
      getInputsOrBuilderList() {
    return inputs_;
  }
  /**
   * <code>repeated .substrait.Rel inputs = 2;</code>
   */
  @java.lang.Override
  public int getInputsCount() {
    return inputs_.size();
  }
  /**
   * <code>repeated .substrait.Rel inputs = 2;</code>
   */
  @java.lang.Override
  public io.substrait.proto.Rel getInputs(int index) {
    return inputs_.get(index);
  }
  /**
   * <code>repeated .substrait.Rel inputs = 2;</code>
   */
  @java.lang.Override
  public io.substrait.proto.RelOrBuilder getInputsOrBuilder(
      int index) {
    return inputs_.get(index);
  }

  public static final int DETAIL_FIELD_NUMBER = 3;
  private com.google.protobuf.Any detail_;
  /**
   * <code>.google.protobuf.Any detail = 3;</code>
   * @return Whether the detail field is set.
   */
  @java.lang.Override
  public boolean hasDetail() {
    return detail_ != null;
  }
  /**
   * <code>.google.protobuf.Any detail = 3;</code>
   * @return The detail.
   */
  @java.lang.Override
  public com.google.protobuf.Any getDetail() {
    return detail_ == null ? com.google.protobuf.Any.getDefaultInstance() : detail_;
  }
  /**
   * <code>.google.protobuf.Any detail = 3;</code>
   */
  @java.lang.Override
  public com.google.protobuf.AnyOrBuilder getDetailOrBuilder() {
    return getDetail();
  }

  private byte memoizedIsInitialized = -1;
  @java.lang.Override
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  @java.lang.Override
  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (common_ != null) {
      output.writeMessage(1, getCommon());
    }
    for (int i = 0; i < inputs_.size(); i++) {
      output.writeMessage(2, inputs_.get(i));
    }
    if (detail_ != null) {
      output.writeMessage(3, getDetail());
    }
    unknownFields.writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (common_ != null) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, getCommon());
    }
    for (int i = 0; i < inputs_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, inputs_.get(i));
    }
    if (detail_ != null) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(3, getDetail());
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof io.substrait.proto.ExtensionMultiRel)) {
      return super.equals(obj);
    }
    io.substrait.proto.ExtensionMultiRel other = (io.substrait.proto.ExtensionMultiRel) obj;

    if (hasCommon() != other.hasCommon()) return false;
    if (hasCommon()) {
      if (!getCommon()
          .equals(other.getCommon())) return false;
    }
    if (!getInputsList()
        .equals(other.getInputsList())) return false;
    if (hasDetail() != other.hasDetail()) return false;
    if (hasDetail()) {
      if (!getDetail()
          .equals(other.getDetail())) return false;
    }
    if (!unknownFields.equals(other.unknownFields)) return false;
    return true;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    if (hasCommon()) {
      hash = (37 * hash) + COMMON_FIELD_NUMBER;
      hash = (53 * hash) + getCommon().hashCode();
    }
    if (getInputsCount() > 0) {
      hash = (37 * hash) + INPUTS_FIELD_NUMBER;
      hash = (53 * hash) + getInputsList().hashCode();
    }
    if (hasDetail()) {
      hash = (37 * hash) + DETAIL_FIELD_NUMBER;
      hash = (53 * hash) + getDetail().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static io.substrait.proto.ExtensionMultiRel parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static io.substrait.proto.ExtensionMultiRel parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static io.substrait.proto.ExtensionMultiRel parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static io.substrait.proto.ExtensionMultiRel parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static io.substrait.proto.ExtensionMultiRel parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static io.substrait.proto.ExtensionMultiRel parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static io.substrait.proto.ExtensionMultiRel parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static io.substrait.proto.ExtensionMultiRel parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static io.substrait.proto.ExtensionMultiRel parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static io.substrait.proto.ExtensionMultiRel parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static io.substrait.proto.ExtensionMultiRel parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static io.substrait.proto.ExtensionMultiRel parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  @java.lang.Override
  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(io.substrait.proto.ExtensionMultiRel prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  @java.lang.Override
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * <pre>
   * Stub to support extension with multiple inputs
   * </pre>
   *
   * Protobuf type {@code substrait.ExtensionMultiRel}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:substrait.ExtensionMultiRel)
      io.substrait.proto.ExtensionMultiRelOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return io.substrait.proto.Relations.internal_static_substrait_ExtensionMultiRel_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return io.substrait.proto.Relations.internal_static_substrait_ExtensionMultiRel_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              io.substrait.proto.ExtensionMultiRel.class, io.substrait.proto.ExtensionMultiRel.Builder.class);
    }

    // Construct using io.substrait.proto.ExtensionMultiRel.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
        getInputsFieldBuilder();
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      if (commonBuilder_ == null) {
        common_ = null;
      } else {
        common_ = null;
        commonBuilder_ = null;
      }
      if (inputsBuilder_ == null) {
        inputs_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
      } else {
        inputsBuilder_.clear();
      }
      if (detailBuilder_ == null) {
        detail_ = null;
      } else {
        detail_ = null;
        detailBuilder_ = null;
      }
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return io.substrait.proto.Relations.internal_static_substrait_ExtensionMultiRel_descriptor;
    }

    @java.lang.Override
    public io.substrait.proto.ExtensionMultiRel getDefaultInstanceForType() {
      return io.substrait.proto.ExtensionMultiRel.getDefaultInstance();
    }

    @java.lang.Override
    public io.substrait.proto.ExtensionMultiRel build() {
      io.substrait.proto.ExtensionMultiRel result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public io.substrait.proto.ExtensionMultiRel buildPartial() {
      io.substrait.proto.ExtensionMultiRel result = new io.substrait.proto.ExtensionMultiRel(this);
      int from_bitField0_ = bitField0_;
      if (commonBuilder_ == null) {
        result.common_ = common_;
      } else {
        result.common_ = commonBuilder_.build();
      }
      if (inputsBuilder_ == null) {
        if (((bitField0_ & 0x00000001) != 0)) {
          inputs_ = java.util.Collections.unmodifiableList(inputs_);
          bitField0_ = (bitField0_ & ~0x00000001);
        }
        result.inputs_ = inputs_;
      } else {
        result.inputs_ = inputsBuilder_.build();
      }
      if (detailBuilder_ == null) {
        result.detail_ = detail_;
      } else {
        result.detail_ = detailBuilder_.build();
      }
      onBuilt();
      return result;
    }

    @java.lang.Override
    public Builder clone() {
      return super.clone();
    }
    @java.lang.Override
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.setField(field, value);
    }
    @java.lang.Override
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return super.clearField(field);
    }
    @java.lang.Override
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return super.clearOneof(oneof);
    }
    @java.lang.Override
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, java.lang.Object value) {
      return super.setRepeatedField(field, index, value);
    }
    @java.lang.Override
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.addRepeatedField(field, value);
    }
    @java.lang.Override
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof io.substrait.proto.ExtensionMultiRel) {
        return mergeFrom((io.substrait.proto.ExtensionMultiRel)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(io.substrait.proto.ExtensionMultiRel other) {
      if (other == io.substrait.proto.ExtensionMultiRel.getDefaultInstance()) return this;
      if (other.hasCommon()) {
        mergeCommon(other.getCommon());
      }
      if (inputsBuilder_ == null) {
        if (!other.inputs_.isEmpty()) {
          if (inputs_.isEmpty()) {
            inputs_ = other.inputs_;
            bitField0_ = (bitField0_ & ~0x00000001);
          } else {
            ensureInputsIsMutable();
            inputs_.addAll(other.inputs_);
          }
          onChanged();
        }
      } else {
        if (!other.inputs_.isEmpty()) {
          if (inputsBuilder_.isEmpty()) {
            inputsBuilder_.dispose();
            inputsBuilder_ = null;
            inputs_ = other.inputs_;
            bitField0_ = (bitField0_ & ~0x00000001);
            inputsBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getInputsFieldBuilder() : null;
          } else {
            inputsBuilder_.addAllMessages(other.inputs_);
          }
        }
      }
      if (other.hasDetail()) {
        mergeDetail(other.getDetail());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    @java.lang.Override
    public final boolean isInitialized() {
      return true;
    }

    @java.lang.Override
    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      io.substrait.proto.ExtensionMultiRel parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (io.substrait.proto.ExtensionMultiRel) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private io.substrait.proto.RelCommon common_;
    private com.google.protobuf.SingleFieldBuilderV3<
        io.substrait.proto.RelCommon, io.substrait.proto.RelCommon.Builder, io.substrait.proto.RelCommonOrBuilder> commonBuilder_;
    /**
     * <code>.substrait.RelCommon common = 1;</code>
     * @return Whether the common field is set.
     */
    public boolean hasCommon() {
      return commonBuilder_ != null || common_ != null;
    }
    /**
     * <code>.substrait.RelCommon common = 1;</code>
     * @return The common.
     */
    public io.substrait.proto.RelCommon getCommon() {
      if (commonBuilder_ == null) {
        return common_ == null ? io.substrait.proto.RelCommon.getDefaultInstance() : common_;
      } else {
        return commonBuilder_.getMessage();
      }
    }
    /**
     * <code>.substrait.RelCommon common = 1;</code>
     */
    public Builder setCommon(io.substrait.proto.RelCommon value) {
      if (commonBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        common_ = value;
        onChanged();
      } else {
        commonBuilder_.setMessage(value);
      }

      return this;
    }
    /**
     * <code>.substrait.RelCommon common = 1;</code>
     */
    public Builder setCommon(
        io.substrait.proto.RelCommon.Builder builderForValue) {
      if (commonBuilder_ == null) {
        common_ = builderForValue.build();
        onChanged();
      } else {
        commonBuilder_.setMessage(builderForValue.build());
      }

      return this;
    }
    /**
     * <code>.substrait.RelCommon common = 1;</code>
     */
    public Builder mergeCommon(io.substrait.proto.RelCommon value) {
      if (commonBuilder_ == null) {
        if (common_ != null) {
          common_ =
            io.substrait.proto.RelCommon.newBuilder(common_).mergeFrom(value).buildPartial();
        } else {
          common_ = value;
        }
        onChanged();
      } else {
        commonBuilder_.mergeFrom(value);
      }

      return this;
    }
    /**
     * <code>.substrait.RelCommon common = 1;</code>
     */
    public Builder clearCommon() {
      if (commonBuilder_ == null) {
        common_ = null;
        onChanged();
      } else {
        common_ = null;
        commonBuilder_ = null;
      }

      return this;
    }
    /**
     * <code>.substrait.RelCommon common = 1;</code>
     */
    public io.substrait.proto.RelCommon.Builder getCommonBuilder() {
      
      onChanged();
      return getCommonFieldBuilder().getBuilder();
    }
    /**
     * <code>.substrait.RelCommon common = 1;</code>
     */
    public io.substrait.proto.RelCommonOrBuilder getCommonOrBuilder() {
      if (commonBuilder_ != null) {
        return commonBuilder_.getMessageOrBuilder();
      } else {
        return common_ == null ?
            io.substrait.proto.RelCommon.getDefaultInstance() : common_;
      }
    }
    /**
     * <code>.substrait.RelCommon common = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        io.substrait.proto.RelCommon, io.substrait.proto.RelCommon.Builder, io.substrait.proto.RelCommonOrBuilder> 
        getCommonFieldBuilder() {
      if (commonBuilder_ == null) {
        commonBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            io.substrait.proto.RelCommon, io.substrait.proto.RelCommon.Builder, io.substrait.proto.RelCommonOrBuilder>(
                getCommon(),
                getParentForChildren(),
                isClean());
        common_ = null;
      }
      return commonBuilder_;
    }

    private java.util.List<io.substrait.proto.Rel> inputs_ =
      java.util.Collections.emptyList();
    private void ensureInputsIsMutable() {
      if (!((bitField0_ & 0x00000001) != 0)) {
        inputs_ = new java.util.ArrayList<io.substrait.proto.Rel>(inputs_);
        bitField0_ |= 0x00000001;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        io.substrait.proto.Rel, io.substrait.proto.Rel.Builder, io.substrait.proto.RelOrBuilder> inputsBuilder_;

    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public java.util.List<io.substrait.proto.Rel> getInputsList() {
      if (inputsBuilder_ == null) {
        return java.util.Collections.unmodifiableList(inputs_);
      } else {
        return inputsBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public int getInputsCount() {
      if (inputsBuilder_ == null) {
        return inputs_.size();
      } else {
        return inputsBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public io.substrait.proto.Rel getInputs(int index) {
      if (inputsBuilder_ == null) {
        return inputs_.get(index);
      } else {
        return inputsBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public Builder setInputs(
        int index, io.substrait.proto.Rel value) {
      if (inputsBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureInputsIsMutable();
        inputs_.set(index, value);
        onChanged();
      } else {
        inputsBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public Builder setInputs(
        int index, io.substrait.proto.Rel.Builder builderForValue) {
      if (inputsBuilder_ == null) {
        ensureInputsIsMutable();
        inputs_.set(index, builderForValue.build());
        onChanged();
      } else {
        inputsBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public Builder addInputs(io.substrait.proto.Rel value) {
      if (inputsBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureInputsIsMutable();
        inputs_.add(value);
        onChanged();
      } else {
        inputsBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public Builder addInputs(
        int index, io.substrait.proto.Rel value) {
      if (inputsBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureInputsIsMutable();
        inputs_.add(index, value);
        onChanged();
      } else {
        inputsBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public Builder addInputs(
        io.substrait.proto.Rel.Builder builderForValue) {
      if (inputsBuilder_ == null) {
        ensureInputsIsMutable();
        inputs_.add(builderForValue.build());
        onChanged();
      } else {
        inputsBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public Builder addInputs(
        int index, io.substrait.proto.Rel.Builder builderForValue) {
      if (inputsBuilder_ == null) {
        ensureInputsIsMutable();
        inputs_.add(index, builderForValue.build());
        onChanged();
      } else {
        inputsBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public Builder addAllInputs(
        java.lang.Iterable<? extends io.substrait.proto.Rel> values) {
      if (inputsBuilder_ == null) {
        ensureInputsIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, inputs_);
        onChanged();
      } else {
        inputsBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public Builder clearInputs() {
      if (inputsBuilder_ == null) {
        inputs_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
        onChanged();
      } else {
        inputsBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public Builder removeInputs(int index) {
      if (inputsBuilder_ == null) {
        ensureInputsIsMutable();
        inputs_.remove(index);
        onChanged();
      } else {
        inputsBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public io.substrait.proto.Rel.Builder getInputsBuilder(
        int index) {
      return getInputsFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public io.substrait.proto.RelOrBuilder getInputsOrBuilder(
        int index) {
      if (inputsBuilder_ == null) {
        return inputs_.get(index);  } else {
        return inputsBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public java.util.List<? extends io.substrait.proto.RelOrBuilder> 
         getInputsOrBuilderList() {
      if (inputsBuilder_ != null) {
        return inputsBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(inputs_);
      }
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public io.substrait.proto.Rel.Builder addInputsBuilder() {
      return getInputsFieldBuilder().addBuilder(
          io.substrait.proto.Rel.getDefaultInstance());
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public io.substrait.proto.Rel.Builder addInputsBuilder(
        int index) {
      return getInputsFieldBuilder().addBuilder(
          index, io.substrait.proto.Rel.getDefaultInstance());
    }
    /**
     * <code>repeated .substrait.Rel inputs = 2;</code>
     */
    public java.util.List<io.substrait.proto.Rel.Builder> 
         getInputsBuilderList() {
      return getInputsFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        io.substrait.proto.Rel, io.substrait.proto.Rel.Builder, io.substrait.proto.RelOrBuilder> 
        getInputsFieldBuilder() {
      if (inputsBuilder_ == null) {
        inputsBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            io.substrait.proto.Rel, io.substrait.proto.Rel.Builder, io.substrait.proto.RelOrBuilder>(
                inputs_,
                ((bitField0_ & 0x00000001) != 0),
                getParentForChildren(),
                isClean());
        inputs_ = null;
      }
      return inputsBuilder_;
    }

    private com.google.protobuf.Any detail_;
    private com.google.protobuf.SingleFieldBuilderV3<
        com.google.protobuf.Any, com.google.protobuf.Any.Builder, com.google.protobuf.AnyOrBuilder> detailBuilder_;
    /**
     * <code>.google.protobuf.Any detail = 3;</code>
     * @return Whether the detail field is set.
     */
    public boolean hasDetail() {
      return detailBuilder_ != null || detail_ != null;
    }
    /**
     * <code>.google.protobuf.Any detail = 3;</code>
     * @return The detail.
     */
    public com.google.protobuf.Any getDetail() {
      if (detailBuilder_ == null) {
        return detail_ == null ? com.google.protobuf.Any.getDefaultInstance() : detail_;
      } else {
        return detailBuilder_.getMessage();
      }
    }
    /**
     * <code>.google.protobuf.Any detail = 3;</code>
     */
    public Builder setDetail(com.google.protobuf.Any value) {
      if (detailBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        detail_ = value;
        onChanged();
      } else {
        detailBuilder_.setMessage(value);
      }

      return this;
    }
    /**
     * <code>.google.protobuf.Any detail = 3;</code>
     */
    public Builder setDetail(
        com.google.protobuf.Any.Builder builderForValue) {
      if (detailBuilder_ == null) {
        detail_ = builderForValue.build();
        onChanged();
      } else {
        detailBuilder_.setMessage(builderForValue.build());
      }

      return this;
    }
    /**
     * <code>.google.protobuf.Any detail = 3;</code>
     */
    public Builder mergeDetail(com.google.protobuf.Any value) {
      if (detailBuilder_ == null) {
        if (detail_ != null) {
          detail_ =
            com.google.protobuf.Any.newBuilder(detail_).mergeFrom(value).buildPartial();
        } else {
          detail_ = value;
        }
        onChanged();
      } else {
        detailBuilder_.mergeFrom(value);
      }

      return this;
    }
    /**
     * <code>.google.protobuf.Any detail = 3;</code>
     */
    public Builder clearDetail() {
      if (detailBuilder_ == null) {
        detail_ = null;
        onChanged();
      } else {
        detail_ = null;
        detailBuilder_ = null;
      }

      return this;
    }
    /**
     * <code>.google.protobuf.Any detail = 3;</code>
     */
    public com.google.protobuf.Any.Builder getDetailBuilder() {
      
      onChanged();
      return getDetailFieldBuilder().getBuilder();
    }
    /**
     * <code>.google.protobuf.Any detail = 3;</code>
     */
    public com.google.protobuf.AnyOrBuilder getDetailOrBuilder() {
      if (detailBuilder_ != null) {
        return detailBuilder_.getMessageOrBuilder();
      } else {
        return detail_ == null ?
            com.google.protobuf.Any.getDefaultInstance() : detail_;
      }
    }
    /**
     * <code>.google.protobuf.Any detail = 3;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        com.google.protobuf.Any, com.google.protobuf.Any.Builder, com.google.protobuf.AnyOrBuilder> 
        getDetailFieldBuilder() {
      if (detailBuilder_ == null) {
        detailBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            com.google.protobuf.Any, com.google.protobuf.Any.Builder, com.google.protobuf.AnyOrBuilder>(
                getDetail(),
                getParentForChildren(),
                isClean());
        detail_ = null;
      }
      return detailBuilder_;
    }
    @java.lang.Override
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    @java.lang.Override
    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:substrait.ExtensionMultiRel)
  }

  // @@protoc_insertion_point(class_scope:substrait.ExtensionMultiRel)
  private static final io.substrait.proto.ExtensionMultiRel DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new io.substrait.proto.ExtensionMultiRel();
  }

  public static io.substrait.proto.ExtensionMultiRel getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<ExtensionMultiRel>
      PARSER = new com.google.protobuf.AbstractParser<ExtensionMultiRel>() {
    @java.lang.Override
    public ExtensionMultiRel parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new ExtensionMultiRel(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ExtensionMultiRel> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ExtensionMultiRel> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public io.substrait.proto.ExtensionMultiRel getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

