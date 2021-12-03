package com.intel.oap.substrait.type;

import io.substrait.Type;

import java.io.Serializable;

public class DateTypeNode implements TypeNode, Serializable {
    private final String name;
    private final Boolean nullable;

    DateTypeNode(String name, Boolean nullable) {
        this.name = name;
        this.nullable = nullable;
    }

    @Override
    public Type toProtobuf() {
        Type.Variation.Builder variationBuilder = Type.Variation.newBuilder();
        variationBuilder.setName(name);

        Type.Date.Builder dateBuilder = Type.Date.newBuilder();
        dateBuilder.setVariation(variationBuilder.build());
        if (nullable) {
            dateBuilder.setNullability(Type.Nullability.NULLABLE);
        } else {
            dateBuilder.setNullability(Type.Nullability.REQUIRED);
        }
        Type.Builder builder = Type.newBuilder();
        builder.setDate(dateBuilder.build());
        return builder.build();
    }
}
