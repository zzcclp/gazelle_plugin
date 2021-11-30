package com.intel.oap.ch.jni;

import io.kyligence.jni.engine.LocalEngine;
import org.apache.commons.io.IOUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.charset.StandardCharsets;

public class TestCHLocalEngine {

    @Before
    public void setup() {
        System.load("/home/myubuntu/Works/c_cpp_projects/Kyligence-ClickHouse/cmake-build-debug/utils/local-engine/liblocal_engine_jnid.so");

    }

    @Test
    public void testLocalEngine() throws Exception{
        String plan = IOUtils.resourceToString("/plan.txt", StandardCharsets.UTF_8);
        LocalEngine localEngine = new LocalEngine(plan.getBytes(StandardCharsets.UTF_8));
        localEngine.execute();
        Assert.assertTrue(localEngine.hasNext());
        byte[] data = localEngine.next();
        Assert.assertEquals(7106, data.length);
    }
}
