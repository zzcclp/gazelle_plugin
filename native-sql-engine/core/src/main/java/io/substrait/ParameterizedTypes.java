// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: parameterized_types.proto

package io.substrait;

public final class ParameterizedTypes {
  private ParameterizedTypes() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_TypeParameter_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_TypeParameter_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_IntegerParameter_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_IntegerParameter_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_NullableInteger_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_NullableInteger_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_ParameterizedFixedChar_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_ParameterizedFixedChar_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_ParameterizedVarChar_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_ParameterizedVarChar_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_ParameterizedFixedBinary_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_ParameterizedFixedBinary_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_ParameterizedDecimal_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_ParameterizedDecimal_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_ParameterizedStruct_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_ParameterizedStruct_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_ParameterizedNamedStruct_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_ParameterizedNamedStruct_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_ParameterizedList_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_ParameterizedList_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_ParameterizedMap_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_ParameterizedMap_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_io_substrait_ParameterizedType_IntegerOption_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_io_substrait_ParameterizedType_IntegerOption_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\031parameterized_types.proto\022\014io.substrai" +
      "t\032\ntype.proto\032\020extensions.proto\"\363\031\n\021Para" +
      "meterizedType\022*\n\004bool\030\001 \001(\0132\032.io.substra" +
      "it.Type.BooleanH\000\022#\n\002i8\030\002 \001(\0132\025.io.subst" +
      "rait.Type.I8H\000\022%\n\003i16\030\003 \001(\0132\026.io.substra" +
      "it.Type.I16H\000\022%\n\003i32\030\005 \001(\0132\026.io.substrai" +
      "t.Type.I32H\000\022%\n\003i64\030\007 \001(\0132\026.io.substrait" +
      ".Type.I64H\000\022\'\n\004fp32\030\n \001(\0132\027.io.substrait" +
      ".Type.FP32H\000\022\'\n\004fp64\030\013 \001(\0132\027.io.substrai" +
      "t.Type.FP64H\000\022+\n\006string\030\014 \001(\0132\031.io.subst" +
      "rait.Type.StringH\000\022+\n\006binary\030\r \001(\0132\031.io." +
      "substrait.Type.BinaryH\000\0221\n\ttimestamp\030\016 \001" +
      "(\0132\034.io.substrait.Type.TimestampH\000\022\'\n\004da" +
      "te\030\020 \001(\0132\027.io.substrait.Type.DateH\000\022\'\n\004t" +
      "ime\030\021 \001(\0132\027.io.substrait.Type.TimeH\000\0228\n\r" +
      "interval_year\030\023 \001(\0132\037.io.substrait.Type." +
      "IntervalYearH\000\0226\n\014interval_day\030\024 \001(\0132\036.i" +
      "o.substrait.Type.IntervalDayH\000\0226\n\014timest" +
      "amp_tz\030\035 \001(\0132\036.io.substrait.Type.Timesta" +
      "mpTZH\000\022\'\n\004uuid\030  \001(\0132\027.io.substrait.Type" +
      ".UUIDH\000\022L\n\nfixed_char\030\025 \001(\01326.io.substra" +
      "it.ParameterizedType.ParameterizedFixedC" +
      "harH\000\022G\n\007varchar\030\026 \001(\01324.io.substrait.Pa" +
      "rameterizedType.ParameterizedVarCharH\000\022P" +
      "\n\014fixed_binary\030\027 \001(\01328.io.substrait.Para" +
      "meterizedType.ParameterizedFixedBinaryH\000" +
      "\022G\n\007decimal\030\030 \001(\01324.io.substrait.Paramet" +
      "erizedType.ParameterizedDecimalH\000\022E\n\006str" +
      "uct\030\031 \001(\01323.io.substrait.ParameterizedTy" +
      "pe.ParameterizedStructH\000\022A\n\004list\030\033 \001(\01321" +
      ".io.substrait.ParameterizedType.Paramete" +
      "rizedListH\000\022?\n\003map\030\034 \001(\01320.io.substrait." +
      "ParameterizedType.ParameterizedMapH\000\0227\n\014" +
      "user_defined\030\037 \001(\0132\037.io.substrait.Extens" +
      "ions.TypeIdH\000\022G\n\016type_parameter\030! \001(\0132-." +
      "io.substrait.ParameterizedType.TypeParam" +
      "eterH\000\032N\n\rTypeParameter\022\014\n\004name\030\001 \001(\t\022/\n" +
      "\006bounds\030\002 \003(\0132\037.io.substrait.Parameteriz" +
      "edType\032\276\001\n\020IntegerParameter\022\014\n\004name\030\001 \001(" +
      "\t\022N\n\025range_start_inclusive\030\002 \001(\0132/.io.su" +
      "bstrait.ParameterizedType.NullableIntege" +
      "r\022L\n\023range_end_exclusive\030\003 \001(\0132/.io.subs" +
      "trait.ParameterizedType.NullableInteger\032" +
      " \n\017NullableInteger\022\r\n\005value\030\001 \001(\003\032\275\001\n\026Pa" +
      "rameterizedFixedChar\022=\n\006length\030\001 \001(\0132-.i" +
      "o.substrait.ParameterizedType.IntegerOpt" +
      "ion\022/\n\tvariation\030\002 \001(\0132\034.io.substrait.Ty" +
      "pe.Variation\0223\n\013nullability\030\003 \001(\0162\036.io.s" +
      "ubstrait.Type.Nullability\032\273\001\n\024Parameteri" +
      "zedVarChar\022=\n\006length\030\001 \001(\0132-.io.substrai" +
      "t.ParameterizedType.IntegerOption\022/\n\tvar" +
      "iation\030\002 \001(\0132\034.io.substrait.Type.Variati" +
      "on\0223\n\013nullability\030\003 \001(\0162\036.io.substrait.T" +
      "ype.Nullability\032\277\001\n\030ParameterizedFixedBi" +
      "nary\022=\n\006length\030\001 \001(\0132-.io.substrait.Para" +
      "meterizedType.IntegerOption\022/\n\tvariation" +
      "\030\002 \001(\0132\034.io.substrait.Type.Variation\0223\n\013" +
      "nullability\030\003 \001(\0162\036.io.substrait.Type.Nu" +
      "llability\032\374\001\n\024ParameterizedDecimal\022<\n\005sc" +
      "ale\030\001 \001(\0132-.io.substrait.ParameterizedTy" +
      "pe.IntegerOption\022@\n\tprecision\030\002 \001(\0132-.io" +
      ".substrait.ParameterizedType.IntegerOpti" +
      "on\022/\n\tvariation\030\003 \001(\0132\034.io.substrait.Typ" +
      "e.Variation\0223\n\013nullability\030\004 \001(\0162\036.io.su" +
      "bstrait.Type.Nullability\032\253\001\n\023Parameteriz" +
      "edStruct\022.\n\005types\030\001 \003(\0132\037.io.substrait.P" +
      "arameterizedType\022/\n\tvariation\030\002 \001(\0132\034.io" +
      ".substrait.Type.Variation\0223\n\013nullability" +
      "\030\003 \001(\0162\036.io.substrait.Type.Nullability\032n" +
      "\n\030ParameterizedNamedStruct\022\r\n\005names\030\001 \003(" +
      "\t\022C\n\006struct\030\002 \001(\01323.io.substrait.Paramet" +
      "erizedType.ParameterizedStruct\032\250\001\n\021Param" +
      "eterizedList\022-\n\004type\030\001 \001(\0132\037.io.substrai" +
      "t.ParameterizedType\022/\n\tvariation\030\002 \001(\0132\034" +
      ".io.substrait.Type.Variation\0223\n\013nullabil" +
      "ity\030\003 \001(\0162\036.io.substrait.Type.Nullabilit" +
      "y\032\326\001\n\020ParameterizedMap\022,\n\003key\030\001 \001(\0132\037.io" +
      ".substrait.ParameterizedType\022.\n\005value\030\002 " +
      "\001(\0132\037.io.substrait.ParameterizedType\022/\n\t" +
      "variation\030\003 \001(\0132\034.io.substrait.Type.Vari" +
      "ation\0223\n\013nullability\030\004 \001(\0162\036.io.substrai" +
      "t.Type.Nullability\032y\n\rIntegerOption\022\021\n\007l" +
      "iteral\030\001 \001(\005H\000\022E\n\tparameter\030\002 \001(\01320.io.s" +
      "ubstrait.ParameterizedType.IntegerParame" +
      "terH\000B\016\n\014integer_typeB\006\n\004kindB\027P\001\252\002\022Subs" +
      "trait.Protobufb\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          io.substrait.TypeOuterClass.getDescriptor(),
          io.substrait.ExtensionsOuterClass.getDescriptor(),
        });
    internal_static_io_substrait_ParameterizedType_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_io_substrait_ParameterizedType_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_descriptor,
        new java.lang.String[] { "Bool", "I8", "I16", "I32", "I64", "Fp32", "Fp64", "String", "Binary", "Timestamp", "Date", "Time", "IntervalYear", "IntervalDay", "TimestampTz", "Uuid", "FixedChar", "Varchar", "FixedBinary", "Decimal", "Struct", "List", "Map", "UserDefined", "TypeParameter", "Kind", });
    internal_static_io_substrait_ParameterizedType_TypeParameter_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(0);
    internal_static_io_substrait_ParameterizedType_TypeParameter_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_TypeParameter_descriptor,
        new java.lang.String[] { "Name", "Bounds", });
    internal_static_io_substrait_ParameterizedType_IntegerParameter_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(1);
    internal_static_io_substrait_ParameterizedType_IntegerParameter_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_IntegerParameter_descriptor,
        new java.lang.String[] { "Name", "RangeStartInclusive", "RangeEndExclusive", });
    internal_static_io_substrait_ParameterizedType_NullableInteger_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(2);
    internal_static_io_substrait_ParameterizedType_NullableInteger_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_NullableInteger_descriptor,
        new java.lang.String[] { "Value", });
    internal_static_io_substrait_ParameterizedType_ParameterizedFixedChar_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(3);
    internal_static_io_substrait_ParameterizedType_ParameterizedFixedChar_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_ParameterizedFixedChar_descriptor,
        new java.lang.String[] { "Length", "Variation", "Nullability", });
    internal_static_io_substrait_ParameterizedType_ParameterizedVarChar_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(4);
    internal_static_io_substrait_ParameterizedType_ParameterizedVarChar_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_ParameterizedVarChar_descriptor,
        new java.lang.String[] { "Length", "Variation", "Nullability", });
    internal_static_io_substrait_ParameterizedType_ParameterizedFixedBinary_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(5);
    internal_static_io_substrait_ParameterizedType_ParameterizedFixedBinary_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_ParameterizedFixedBinary_descriptor,
        new java.lang.String[] { "Length", "Variation", "Nullability", });
    internal_static_io_substrait_ParameterizedType_ParameterizedDecimal_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(6);
    internal_static_io_substrait_ParameterizedType_ParameterizedDecimal_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_ParameterizedDecimal_descriptor,
        new java.lang.String[] { "Scale", "Precision", "Variation", "Nullability", });
    internal_static_io_substrait_ParameterizedType_ParameterizedStruct_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(7);
    internal_static_io_substrait_ParameterizedType_ParameterizedStruct_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_ParameterizedStruct_descriptor,
        new java.lang.String[] { "Types", "Variation", "Nullability", });
    internal_static_io_substrait_ParameterizedType_ParameterizedNamedStruct_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(8);
    internal_static_io_substrait_ParameterizedType_ParameterizedNamedStruct_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_ParameterizedNamedStruct_descriptor,
        new java.lang.String[] { "Names", "Struct", });
    internal_static_io_substrait_ParameterizedType_ParameterizedList_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(9);
    internal_static_io_substrait_ParameterizedType_ParameterizedList_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_ParameterizedList_descriptor,
        new java.lang.String[] { "Type", "Variation", "Nullability", });
    internal_static_io_substrait_ParameterizedType_ParameterizedMap_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(10);
    internal_static_io_substrait_ParameterizedType_ParameterizedMap_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_ParameterizedMap_descriptor,
        new java.lang.String[] { "Key", "Value", "Variation", "Nullability", });
    internal_static_io_substrait_ParameterizedType_IntegerOption_descriptor =
      internal_static_io_substrait_ParameterizedType_descriptor.getNestedTypes().get(11);
    internal_static_io_substrait_ParameterizedType_IntegerOption_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_io_substrait_ParameterizedType_IntegerOption_descriptor,
        new java.lang.String[] { "Literal", "Parameter", "IntegerType", });
    io.substrait.TypeOuterClass.getDescriptor();
    io.substrait.ExtensionsOuterClass.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}