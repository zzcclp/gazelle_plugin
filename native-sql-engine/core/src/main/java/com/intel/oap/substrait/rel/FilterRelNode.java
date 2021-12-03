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

package com.intel.oap.substrait.rel;

import com.intel.oap.substrait.expression.ExpressionNode;
import com.intel.oap.substrait.type.TypeNode;
import io.substrait.proto.*;

import java.io.Serializable;
import java.util.ArrayList;

public class FilterRelNode implements RelNode, Serializable {
    private final RelNode input;
    private final ExpressionNode condition;
    private final ArrayList<TypeNode> types = new ArrayList<>();

    FilterRelNode(RelNode input,
                  ExpressionNode condition,
                  ArrayList<TypeNode> types) {
        this.input = input;
        this.condition = condition;
        this.types.addAll(types);
    }

    @Override
    public Rel toProtobuf() {
        FilterRel.Builder filterBuilder = FilterRel.newBuilder();
        if (input != null) {
            filterBuilder.setInput(input.toProtobuf());
        }
        filterBuilder.setCondition(condition.toProtobuf());
        /*for (TypeNode type : types) {
            filterBuilder.addInputTypes(type.toProtobuf());
        }*/
        Rel.Builder builder = Rel.newBuilder();
        builder.setFilter(filterBuilder.build());
        return builder.build();
    }
}
