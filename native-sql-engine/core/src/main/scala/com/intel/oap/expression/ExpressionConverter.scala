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

package com.intel.oap.expression

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.expressions.BindReferences.bindReferences
import org.apache.spark.sql.types.DecimalType
object ExpressionConverter extends Logging {
  def replaceWithExpressionTransformer(
      expr: Expression,
      attributeSeq: Seq[Attribute]): Expression =
    expr match {
      case a: Alias =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = new AliasTransformer(
          replaceWithExpressionTransformer(
            a.child,
            attributeSeq),
          a.name)(a.exprId, a.qualifier, a.explicitMetadata)
        if (!transformer.doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${a.getClass} | ${a} is not currently supported.")
        }
        transformer
      case a: AttributeReference =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        if (attributeSeq == null) {
          throw new UnsupportedOperationException(s"attributeSeq should not be null.")
        }
        val bindReference =
          BindReferences.bindReference(expr, attributeSeq, allowFailures = true)
        if (bindReference == expr) {
          // bind failure
          throw new UnsupportedOperationException(s"attribute binding failed.")
        } else {
          val b = bindReference.asInstanceOf[BoundReference]
          val transformer = new AttributeReferenceTransformer(
            a.name, b.ordinal, a.dataType, a.nullable, a.metadata)(a.exprId, a.qualifier)
          if (!transformer.doValidate()) {
            throw new UnsupportedOperationException(
              s" --> ${a.getClass} | ${a} is not currently supported.")
          }
          transformer
        }
      case lit: Literal =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = new LiteralTransformer(lit)
        if (!transformer.doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${lit.getClass} | ${lit} is not currently supported.")
        }
        transformer
      case binArith: BinaryArithmetic =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = BinaryArithmeticTransformer.create(
          replaceWithExpressionTransformer(
            binArith.left,
            attributeSeq),
          replaceWithExpressionTransformer(
            binArith.right,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${binArith.getClass} | ${binArith} is not currently supported.")
        }
        transformer
      case b: BoundReference =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = new BoundReferenceTransformer(b.ordinal, b.dataType, b.nullable)
        if (!transformer.doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${b.getClass} | ${b} is not currently supported.")
        }
        transformer
      case b: BinaryOperator =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = BinaryOperatorTransformer.create(
          replaceWithExpressionTransformer(
            b.left,
            attributeSeq),
          replaceWithExpressionTransformer(
            b.right,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${b.getClass} | ${b} is not currently supported.")
        }
        transformer
      case b: ShiftLeft =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = BinaryOperatorTransformer.create(
          replaceWithExpressionTransformer(
            b.left,
            attributeSeq),
          replaceWithExpressionTransformer(
            b.right,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${b.getClass} | ${b} is not currently supported.")
        }
        transformer
      case b: ShiftRight =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = BinaryOperatorTransformer.create(
          replaceWithExpressionTransformer(
            b.left,
            attributeSeq),
          replaceWithExpressionTransformer(
            b.right,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${b.getClass} | ${b} is not currently supported.")
        }
        transformer
      case sp: StringPredicate =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = BinaryOperatorTransformer.create(
          replaceWithExpressionTransformer(
            sp.left,
            attributeSeq),
          replaceWithExpressionTransformer(
            sp.right,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${sp.getClass} | ${sp} is not currently supported.")
        }
        transformer
      case sr: StringRegexExpression =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = BinaryOperatorTransformer.create(
          replaceWithExpressionTransformer(
            sr.left,
            attributeSeq),
          replaceWithExpressionTransformer(
            sr.right,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${sr.getClass} | ${sr} is not currently supported.")
        }
        transformer
      case i: If =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = IfOperatorTransformer.create(
          replaceWithExpressionTransformer(
            i.predicate,
            attributeSeq),
          replaceWithExpressionTransformer(
            i.trueValue,
            attributeSeq),
          replaceWithExpressionTransformer(
            i.falseValue,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${i.getClass} | ${i} is not currently supported.")
        }
        transformer
      case cw: CaseWhen =>
        logInfo(s"${expr.getClass} ${expr} is supportedn.")
        val colBranches = cw.branches.map { expr => {(
              replaceWithExpressionTransformer(
                expr._1,
                attributeSeq),
              replaceWithExpressionTransformer(
                expr._2,
                attributeSeq))
          }
        }
        val colElseValue = cw.elseValue.map { expr => {
            replaceWithExpressionTransformer(
              expr,
              attributeSeq)
          }
        }
        logInfo(s"col_branches: $colBranches")
        logInfo(s"col_else: $colElseValue")
        val transformer = CaseWhenOperatorTransformer.create(colBranches, colElseValue, expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${cw.getClass} | ${cw} is not currently supported.")
        }
        transformer
      case c: Coalesce =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val exps = c.children.map { expr =>
          replaceWithExpressionTransformer(
            expr,
            attributeSeq)
        }
        val transformer = CoalesceOperatorTransformer.create(exps, expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${c.getClass} | ${c} is not currently supported.")
        }
        transformer
      case i: In =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = InOperatorTransformer.create(
          replaceWithExpressionTransformer(
            i.value,
            attributeSeq),
          i.list,
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${i.getClass} | ${i} is not currently supported.")
        }
        transformer
      case i: InSet =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = InSetOperatorTransformer.create(
          replaceWithExpressionTransformer(
            i.child,
            attributeSeq),
          i.hset,
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${i.getClass} | ${i} is not currently supported.")
        }
        transformer
      case ss: Substring =>
        logInfo(s"${expr.getClass} ${expr} is supported.")
        val transformer = TernaryOperatorTransformer.create(
          replaceWithExpressionTransformer(
            ss.str,
            attributeSeq),
          replaceWithExpressionTransformer(
            ss.pos,
            attributeSeq),
          replaceWithExpressionTransformer(
            ss.len,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${ss.getClass} | ${ss} is not currently supported.")
        }
        transformer
      case u: UnaryExpression =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        if (!u.isInstanceOf[CheckOverflow] || !u.child.isInstanceOf[Divide]) {
          val transformer = UnaryOperatorTransformer.create(
            replaceWithExpressionTransformer(
              u.child,
              attributeSeq),
            expr)
          if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
            throw new UnsupportedOperationException(
              s" --> ${u.getClass} | ${u} is not currently supported.")
          }
          transformer
        } else {
          // CheckOverflow[Divide]: pass resType to Divide to avoid precision loss
          val divide = u.child.asInstanceOf[Divide]
          val columnarDivide = BinaryArithmeticTransformer.createDivide(
            replaceWithExpressionTransformer(
              divide.left,
              attributeSeq),
            replaceWithExpressionTransformer(
              divide.right,
              attributeSeq),
            divide,
            u.dataType.asInstanceOf[DecimalType])
          if (columnarDivide.asInstanceOf[ExpressionTransformer].doValidate()) {
            throw new UnsupportedOperationException(
              s" --> ${u.getClass} | ${u} is not currently supported.")
          }
          val transformer = UnaryOperatorTransformer.create(
            columnarDivide,
            expr)
          if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
            throw new UnsupportedOperationException(
              s" --> ${u.getClass} | ${u} is not currently supported.")
          }
          transformer
        }
      case oaps: com.intel.oap.expression.ScalarSubqueryTransformer =>
        oaps
      case s: org.apache.spark.sql.execution.ScalarSubquery =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = new ScalarSubqueryTransformer(s)
        if (!transformer.doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${s.getClass} | ${s} is not currently supported.")
        }
        transformer
      case c: Concat =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val exps = c.children.map { expr =>
          replaceWithExpressionTransformer(
            expr,
            attributeSeq)
        }
        val transformer = ConcatOperatorTransformer.create(exps, expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${c.getClass} | ${c} is not currently supported.")
        }
        transformer
      case r: Round =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = RoundOperatorTransformer.create(
          replaceWithExpressionTransformer(
            r.child,
            attributeSeq),
          replaceWithExpressionTransformer(
            r.scale,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${r.getClass} | ${r} is not currently supported.")
        }
        transformer
      case b: BinaryExpression =>
        logInfo(s"${expr.getClass} ${expr} is supported")
        val transformer = BinaryExpressionTransformer.create(
          replaceWithExpressionTransformer(
            b.left,
            attributeSeq),
          replaceWithExpressionTransformer(
            b.right,
            attributeSeq),
          expr)
        if (!transformer.asInstanceOf[ExpressionTransformer].doValidate()) {
          throw new UnsupportedOperationException(
            s" --> ${b.getClass} | ${b} is not currently supported.")
        }
        transformer
      case expr =>
        throw new UnsupportedOperationException(
          s" --> ${expr.getClass} | ${expr} is not currently supported.")
    }
}
