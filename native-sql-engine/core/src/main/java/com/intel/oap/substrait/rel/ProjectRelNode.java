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
import com.intel.oap.substrait.derivation.DerivationExpressionNode;
import com.intel.oap.substrait.type.TypeNode;
import io.substrait.*;

import java.util.ArrayList;

public class ProjectRelNode implements RelNode {
    private final Rel input;
    private final ArrayList<DerivationExpression> derivationExpressions =
            new ArrayList<DerivationExpression>();
    private final ArrayList<Type> inputTypes = new ArrayList<Type>();
    private final ArrayList<Type> outputTypes = new ArrayList<Type>();

    ProjectRelNode(Rel input,
               ArrayList<DerivationExpression> derivationExpressions,
               ArrayList<Type> inputTypes,
               ArrayList<Type> outputTypes) {
        this.input = input;
        this.derivationExpressions.addAll(derivationExpressions);
        this.inputTypes.addAll(inputTypes);
        this.outputTypes.addAll(outputTypes);
    }

    @Override
    public Rel toProtobuf() {
        ProjectRel.Builder projectBuilder =
                ProjectRel.newBuilder();
        projectBuilder.setInput(input);
        for (DerivationExpression expr : derivationExpressions) {
            projectBuilder.addDerivationExpressions(expr);
        }
        for (Type type : inputTypes) {
            projectBuilder.addInputTypes(type);
        }
//        for (Type type : outputTypes) {
//            projectBuilder.addOutputTypes(type);
//        }

        Rel.Builder builder = Rel.newBuilder();
        builder.setProject(projectBuilder.build());
        return builder.build();
    }
}
