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

package org.apache.spark.sql.execution.arrow;

import com.intel.oap.ch.jni.CHJniInstance;
import com.intel.oap.vectorized.ArrowWritableColumnVector;
import io.kyligence.jni.engine.LocalEngine;
import io.kyligence.jni.engine.SparkRowInfo;
import org.apache.arrow.dataset.jni.UnsafeRecordBatchSerializer;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowFileReader;
import org.apache.arrow.vector.ipc.message.ArrowBlock;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;
import org.apache.spark.sql.execution.datasources.v2.arrow.SparkMemoryUtils;
import org.apache.spark.sql.vectorized.ColumnarBatch;

import java.io.IOException;

public class CHBatchIterator implements BasicBatchIterator {

  private LocalEngine localEngine;
  private boolean closed = false;

  public CHBatchIterator() throws IOException {}

  public CHBatchIterator(byte[] plan, String soFilePath) throws IOException {
    this.localEngine = CHJniInstance
      .getInstance(soFilePath)
            .buildLocalEngine(plan);
    this.localEngine.execute();
  }

  @Override
  public boolean hasNext() throws IOException {
    return this.localEngine.hasNext();
  }

  @Override
  public SparkRowInfo next() throws IOException {
    if (this.localEngine == null) {
      return null;
    }
    return this.localEngine.next();
  }

  public ArrowRecordBatch next2() throws IOException {
    BufferAllocator allocator = SparkMemoryUtils.contextAllocator();
    if (this.localEngine == null) {
      return null;
    }
    // return the arrow 'WriteRecordBatchMessage' data
    byte[] serializedRecordBatch = null; //this.localEngine.next();
    if (serializedRecordBatch == null) {
      return null;
    }
    /* ByteArrayInputStream bais = new ByteArrayInputStream(serializedRecordBatch);
    return MessageSerializer.deserializeRecordBatch(new ReadChannel(Channels.newChannel(bais)),
            allocator); */
    return UnsafeRecordBatchSerializer.deserializeUnsafe(allocator,
            serializedRecordBatch);
  }

  public ColumnarBatch next1() throws IOException {
    BufferAllocator allocator = SparkMemoryUtils.contextAllocator();
    if (this.localEngine == null) {
      return null;
    }
    // return the arrow 'ArrowTable' data
    byte[] serializedRecordBatch = null; //this.localEngine.next();
    if (serializedRecordBatch == null) {
      return null;
    }
    ArrowFileReader reader = new ArrowFileReader(
            new ByteArrayReadableSeekableByteChannel(serializedRecordBatch), allocator);
    ArrowBlock block = reader.getRecordBlocks().get(0);
    reader.loadRecordBatch(block);
    VectorSchemaRoot recordBatch = reader.getVectorSchemaRoot();
    ArrowWritableColumnVector[] output = ArrowWritableColumnVector
            .loadColumns(recordBatch.getRowCount(), recordBatch.getFieldVectors());
    ColumnarBatch cb = new ColumnarBatch(output, recordBatch.getRowCount());
    // don't close reader, it will clear all data
    //reader.close(false);
    return cb;
  }

  @Override
  public void close() {
    if (!closed) {
      if (this.localEngine != null) {
        try {
          this.localEngine.close();
        } catch (IOException e) {
        }
      }
      closed = true;
    }
  }
}
