/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.oap.substrait.type;

import io.substrait.*;

public class FP64TypeNode implements TypeNode {
    private final String name;
    private final Boolean nullable;

    FP64TypeNode(String name, Boolean nullable) {
        this.name = name;
        this.nullable = nullable;
    }

    @Override
    public Type toProtobuf() {
        Type.Variation.Builder variationBuilder =
                Type.Variation.newBuilder();
        variationBuilder.setName(name);

        Type.FP64.Builder doubleBuilder = Type.FP64.newBuilder();
        doubleBuilder.setVariation(variationBuilder.build());
        if (nullable) {
            doubleBuilder.setNullability(Type.Nullability.NULLABLE);
        } else {
            doubleBuilder.setNullability(Type.Nullability.REQUIRED);
        }

        Type.Builder builder = Type.newBuilder();
        builder.setFp64(doubleBuilder.build());
        return builder.build();
    }
}