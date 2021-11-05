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

package com.intel.oap.substrait.derivation;

import io.substrait.*;

public class BinaryOPNode implements DerivationExpressionNode {
    private final String op;
    private final DerivationExpression arg1;
    private final DerivationExpression arg2;

    BinaryOPNode(String op, DerivationExpression arg1, DerivationExpression arg2) {
        this.op = op;
        this.arg1 = arg1;
        this.arg2 = arg2;
    }

    @Override
    public DerivationExpression toProtobuf() {
        DerivationExpression.BinaryOp.Builder binaryBuilder =
                DerivationExpression.BinaryOp.newBuilder();
        switch(op){
            case "multiply" :
                binaryBuilder.setOpType(
                        DerivationExpression.BinaryOp.OpType.MULTIPLY);
                break;
            case "divide" :
                binaryBuilder.setOpType(
                        DerivationExpression.BinaryOp.OpType.DIVIDE);
                break;
            default :
                System.out.println("Not supported.");
        }
        binaryBuilder.setArg1(arg1);
        binaryBuilder.setArg2(arg2);

        DerivationExpression.Builder builder =
                DerivationExpression.newBuilder();
        builder.setBinaryOp(binaryBuilder.build());
        return builder.build();
    }
}
