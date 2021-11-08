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

import com.intel.oap.substrait.expression.AggregateFunctionNode;
import io.substrait.AggregateRel;
import io.substrait.Expression;
import io.substrait.Rel;

import java.util.ArrayList;

public class AggregateRelNode implements RelNode {
    private final RelNode input;
    private final ArrayList<Integer> groupings = new ArrayList<>();
    private final ArrayList<AggregateFunctionNode> aggregateFunctionNodes =
            new ArrayList<>();
    private final String phase;

    AggregateRelNode(RelNode input,
                     ArrayList<Integer> groupings,
                     ArrayList<AggregateFunctionNode> aggregateFunctionNodes,
                     String phase) {
        this.input = input;
        this.groupings.addAll(groupings);
        this.aggregateFunctionNodes.addAll(aggregateFunctionNodes);
        this.phase = phase;
    }

    AggregateRelNode(RelNode input,
                     ArrayList<Integer> groupings,
                     ArrayList<AggregateFunctionNode> aggregateFunctionNodes) {
        this.input = input;
        this.groupings.addAll(groupings);
        this.aggregateFunctionNodes.addAll(aggregateFunctionNodes);
        this.phase = null;
    }

    @Override
    public Rel toProtobuf() {
        AggregateRel.Grouping.Builder groupingBuilder =
                AggregateRel.Grouping.newBuilder();
        for (Integer integer : groupings) {
            groupingBuilder.addInputFields(integer.intValue());
        }

        AggregateRel.Builder aggBuilder = AggregateRel.newBuilder();
        aggBuilder.addGroupings(groupingBuilder.build());

        for (AggregateFunctionNode aggregateFunctionNode : aggregateFunctionNodes) {
            AggregateRel.Measure.Builder measureBuilder = AggregateRel.Measure.newBuilder();
            measureBuilder.setMeasure(aggregateFunctionNode.toProtobuf());
            aggBuilder.addMeasures(measureBuilder.build());
        }
        if (input != null) {
            aggBuilder.setInput(input.toProtobuf());
        }
        if (phase != null) {
            switch(phase) {
                case "PARTIAL":
                    aggBuilder.setPhase(Expression.AggregationPhase.INITIAL_TO_INTERMEDIATE);
                    break;
                case "PARTIAL_MERGE":
                    aggBuilder.setPhase(Expression.AggregationPhase.INTERMEDIATE_TO_INTERMEDIATE);
                    break;
                case "FINAL":
                    aggBuilder.setPhase(Expression.AggregationPhase.INTERMEDIATE_TO_RESULT);
                    break;
                default:
                    System.out.println("Not supported.");
            }
        }
        Rel.Builder builder = Rel.newBuilder();
        builder.setAggregate(aggBuilder.build());
        return builder.build();
    }
}
