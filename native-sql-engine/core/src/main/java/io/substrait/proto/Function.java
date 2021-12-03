// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: substrait/function.proto

package io.substrait.proto;

public final class Function {
  private Function() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_FinalArgVariadic_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_FinalArgVariadic_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_FinalArgNormal_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_FinalArgNormal_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_Scalar_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_Scalar_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_Aggregate_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_Aggregate_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_Window_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_Window_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_Description_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_Description_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_Implementation_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_Implementation_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_Argument_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_Argument_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_Argument_ValueArgument_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_Argument_ValueArgument_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_Argument_TypeArgument_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_Argument_TypeArgument_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_substrait_FunctionSignature_Argument_EnumArgument_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_substrait_FunctionSignature_Argument_EnumArgument_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\030substrait/function.proto\022\tsubstrait\032\024s" +
      "ubstrait/type.proto\032#substrait/parameter" +
      "ized_types.proto\032 substrait/type_express" +
      "ions.proto\"\301\025\n\021FunctionSignature\032\235\002\n\020Fin" +
      "alArgVariadic\022\020\n\010min_args\030\001 \001(\003\022\020\n\010max_a" +
      "rgs\030\002 \001(\003\022W\n\013consistency\030\003 \001(\0162B.substra" +
      "it.FunctionSignature.FinalArgVariadic.Pa" +
      "rameterConsistency\"\213\001\n\024ParameterConsiste" +
      "ncy\022%\n!PARAMETER_CONSISTENCY_UNSPECIFIED" +
      "\020\000\022$\n PARAMETER_CONSISTENCY_CONSISTENT\020\001" +
      "\022&\n\"PARAMETER_CONSISTENCY_INCONSISTENT\020\002" +
      "\032\020\n\016FinalArgNormal\032\332\003\n\006Scalar\0228\n\targumen" +
      "ts\030\002 \003(\0132%.substrait.FunctionSignature.A" +
      "rgument\022\014\n\004name\030\003 \003(\t\022=\n\013description\030\004 \001" +
      "(\0132(.substrait.FunctionSignature.Descrip" +
      "tion\022\025\n\rdeterministic\030\007 \001(\010\022\031\n\021session_d" +
      "ependent\030\010 \001(\010\0224\n\013output_type\030\t \001(\0132\037.su" +
      "bstrait.DerivationExpression\022A\n\010variadic" +
      "\030\n \001(\0132-.substrait.FunctionSignature.Fin" +
      "alArgVariadicH\000\022=\n\006normal\030\013 \001(\0132+.substr" +
      "ait.FunctionSignature.FinalArgNormalH\000\022D" +
      "\n\017implementations\030\014 \003(\0132+.substrait.Func" +
      "tionSignature.ImplementationB\031\n\027final_va" +
      "riable_behavior\032\253\004\n\tAggregate\0228\n\targumen" +
      "ts\030\002 \003(\0132%.substrait.FunctionSignature.A" +
      "rgument\022\014\n\004name\030\003 \001(\t\022=\n\013description\030\004 \001" +
      "(\0132(.substrait.FunctionSignature.Descrip" +
      "tion\022\025\n\rdeterministic\030\007 \001(\010\022\031\n\021session_d" +
      "ependent\030\010 \001(\010\0224\n\013output_type\030\t \001(\0132\037.su" +
      "bstrait.DerivationExpression\022A\n\010variadic" +
      "\030\n \001(\0132-.substrait.FunctionSignature.Fin" +
      "alArgVariadicH\000\022=\n\006normal\030\013 \001(\0132+.substr" +
      "ait.FunctionSignature.FinalArgNormalH\000\022\017" +
      "\n\007ordered\030\016 \001(\010\022\017\n\007max_set\030\014 \001(\004\022*\n\021inte" +
      "rmediate_type\030\r \001(\0132\017.substrait.Type\022D\n\017" +
      "implementations\030\017 \003(\0132+.substrait.Functi" +
      "onSignature.ImplementationB\031\n\027final_vari" +
      "able_behavior\032\336\005\n\006Window\0228\n\targuments\030\002 " +
      "\003(\0132%.substrait.FunctionSignature.Argume" +
      "nt\022\014\n\004name\030\003 \003(\t\022=\n\013description\030\004 \001(\0132(." +
      "substrait.FunctionSignature.Description\022" +
      "\025\n\rdeterministic\030\007 \001(\010\022\031\n\021session_depend" +
      "ent\030\010 \001(\010\022:\n\021intermediate_type\030\t \001(\0132\037.s" +
      "ubstrait.DerivationExpression\0224\n\013output_" +
      "type\030\n \001(\0132\037.substrait.DerivationExpress" +
      "ion\022A\n\010variadic\030\020 \001(\0132-.substrait.Functi" +
      "onSignature.FinalArgVariadicH\000\022=\n\006normal" +
      "\030\021 \001(\0132+.substrait.FunctionSignature.Fin" +
      "alArgNormalH\000\022\017\n\007ordered\030\013 \001(\010\022\017\n\007max_se" +
      "t\030\014 \001(\004\022C\n\013window_type\030\016 \001(\0162..substrait" +
      ".FunctionSignature.Window.WindowType\022D\n\017" +
      "implementations\030\017 \003(\0132+.substrait.Functi" +
      "onSignature.Implementation\"_\n\nWindowType" +
      "\022\033\n\027WINDOW_TYPE_UNSPECIFIED\020\000\022\031\n\025WINDOW_" +
      "TYPE_STREAMING\020\001\022\031\n\025WINDOW_TYPE_PARTITIO" +
      "N\020\002B\031\n\027final_variable_behavior\032-\n\013Descri" +
      "ption\022\020\n\010language\030\001 \001(\t\022\014\n\004body\030\002 \001(\t\032\246\001" +
      "\n\016Implementation\022>\n\004type\030\001 \001(\01620.substra" +
      "it.FunctionSignature.Implementation.Type" +
      "\022\013\n\003uri\030\002 \001(\t\"G\n\004Type\022\024\n\020TYPE_UNSPECIFIE" +
      "D\020\000\022\025\n\021TYPE_WEB_ASSEMBLY\020\001\022\022\n\016TYPE_TRINO" +
      "_JAR\020\002\032\265\003\n\010Argument\022\014\n\004name\030\001 \001(\t\022D\n\005val" +
      "ue\030\002 \001(\01323.substrait.FunctionSignature.A" +
      "rgument.ValueArgumentH\000\022B\n\004type\030\003 \001(\01322." +
      "substrait.FunctionSignature.Argument.Typ" +
      "eArgumentH\000\022B\n\004enum\030\004 \001(\01322.substrait.Fu" +
      "nctionSignature.Argument.EnumArgumentH\000\032" +
      "M\n\rValueArgument\022*\n\004type\030\001 \001(\0132\034.substra" +
      "it.ParameterizedType\022\020\n\010constant\030\002 \001(\010\032:" +
      "\n\014TypeArgument\022*\n\004type\030\001 \001(\0132\034.substrait" +
      ".ParameterizedType\0321\n\014EnumArgument\022\017\n\007op" +
      "tions\030\001 \003(\t\022\020\n\010optional\030\002 \001(\010B\017\n\rargumen" +
      "t_kindB+\n\022io.substrait.protoP\001\252\002\022Substra" +
      "it.Protobufb\006proto3"
    };
    descriptor = com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          io.substrait.proto.TypeOuterClass.getDescriptor(),
          io.substrait.proto.ParameterizedTypes.getDescriptor(),
          io.substrait.proto.TypeExpressions.getDescriptor(),
        });
    internal_static_substrait_FunctionSignature_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_substrait_FunctionSignature_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_descriptor,
        new java.lang.String[] { });
    internal_static_substrait_FunctionSignature_FinalArgVariadic_descriptor =
      internal_static_substrait_FunctionSignature_descriptor.getNestedTypes().get(0);
    internal_static_substrait_FunctionSignature_FinalArgVariadic_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_FinalArgVariadic_descriptor,
        new java.lang.String[] { "MinArgs", "MaxArgs", "Consistency", });
    internal_static_substrait_FunctionSignature_FinalArgNormal_descriptor =
      internal_static_substrait_FunctionSignature_descriptor.getNestedTypes().get(1);
    internal_static_substrait_FunctionSignature_FinalArgNormal_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_FinalArgNormal_descriptor,
        new java.lang.String[] { });
    internal_static_substrait_FunctionSignature_Scalar_descriptor =
      internal_static_substrait_FunctionSignature_descriptor.getNestedTypes().get(2);
    internal_static_substrait_FunctionSignature_Scalar_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_Scalar_descriptor,
        new java.lang.String[] { "Arguments", "Name", "Description", "Deterministic", "SessionDependent", "OutputType", "Variadic", "Normal", "Implementations", "FinalVariableBehavior", });
    internal_static_substrait_FunctionSignature_Aggregate_descriptor =
      internal_static_substrait_FunctionSignature_descriptor.getNestedTypes().get(3);
    internal_static_substrait_FunctionSignature_Aggregate_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_Aggregate_descriptor,
        new java.lang.String[] { "Arguments", "Name", "Description", "Deterministic", "SessionDependent", "OutputType", "Variadic", "Normal", "Ordered", "MaxSet", "IntermediateType", "Implementations", "FinalVariableBehavior", });
    internal_static_substrait_FunctionSignature_Window_descriptor =
      internal_static_substrait_FunctionSignature_descriptor.getNestedTypes().get(4);
    internal_static_substrait_FunctionSignature_Window_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_Window_descriptor,
        new java.lang.String[] { "Arguments", "Name", "Description", "Deterministic", "SessionDependent", "IntermediateType", "OutputType", "Variadic", "Normal", "Ordered", "MaxSet", "WindowType", "Implementations", "FinalVariableBehavior", });
    internal_static_substrait_FunctionSignature_Description_descriptor =
      internal_static_substrait_FunctionSignature_descriptor.getNestedTypes().get(5);
    internal_static_substrait_FunctionSignature_Description_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_Description_descriptor,
        new java.lang.String[] { "Language", "Body", });
    internal_static_substrait_FunctionSignature_Implementation_descriptor =
      internal_static_substrait_FunctionSignature_descriptor.getNestedTypes().get(6);
    internal_static_substrait_FunctionSignature_Implementation_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_Implementation_descriptor,
        new java.lang.String[] { "Type", "Uri", });
    internal_static_substrait_FunctionSignature_Argument_descriptor =
      internal_static_substrait_FunctionSignature_descriptor.getNestedTypes().get(7);
    internal_static_substrait_FunctionSignature_Argument_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_Argument_descriptor,
        new java.lang.String[] { "Name", "Value", "Type", "Enum", "ArgumentKind", });
    internal_static_substrait_FunctionSignature_Argument_ValueArgument_descriptor =
      internal_static_substrait_FunctionSignature_Argument_descriptor.getNestedTypes().get(0);
    internal_static_substrait_FunctionSignature_Argument_ValueArgument_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_Argument_ValueArgument_descriptor,
        new java.lang.String[] { "Type", "Constant", });
    internal_static_substrait_FunctionSignature_Argument_TypeArgument_descriptor =
      internal_static_substrait_FunctionSignature_Argument_descriptor.getNestedTypes().get(1);
    internal_static_substrait_FunctionSignature_Argument_TypeArgument_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_Argument_TypeArgument_descriptor,
        new java.lang.String[] { "Type", });
    internal_static_substrait_FunctionSignature_Argument_EnumArgument_descriptor =
      internal_static_substrait_FunctionSignature_Argument_descriptor.getNestedTypes().get(2);
    internal_static_substrait_FunctionSignature_Argument_EnumArgument_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_substrait_FunctionSignature_Argument_EnumArgument_descriptor,
        new java.lang.String[] { "Options", "Optional", });
    io.substrait.proto.TypeOuterClass.getDescriptor();
    io.substrait.proto.ParameterizedTypes.getDescriptor();
    io.substrait.proto.TypeExpressions.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
