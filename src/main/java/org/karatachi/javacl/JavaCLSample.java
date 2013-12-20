package org.karatachi.javacl;

import java.io.IOException;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.CLPlatform.ContextProperties;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.util.IOUtils;

public class JavaCLSample {
    private static final int BUFFER_SIZE = 100 * 1024;
    private static final int LOOP_COUNT = 10;

    public static void main(String[] args) throws Exception {
        List<CLDevice> devices = getDevices();

        System.out.println("====");
        System.out.println("Reference");
        benchRefernece();

        for (CLDevice device : devices) {
            CLContext context =
                    JavaCL.createContext(
                            Collections.<ContextProperties, Object> emptyMap(),
                            device);

            printDeviceInfo(context);
            bench(context);
        }
    }

    private static List<CLDevice> getDevices() {
        List<CLDevice> devices = new ArrayList<>();
        for (CLPlatform platform : JavaCL.listPlatforms()) {
            for (CLDevice device : platform.listAllDevices(true)) {
                devices.add(device);
            }
        }
        return devices;
    }

    private static void printDeviceInfo(CLContext context) {
        System.out.println("====");
        System.out.println("Platform: " + context.getPlatform().getName());
        System.out.println("Device: " + context.getDevices()[0].getName());
        System.out.println("MaxMemAllocSize: "
                + String.format("%,d", context.getMaxMemAllocSize()));
    }

    private static void benchRefernece() {
        float[] in = new float[BUFFER_SIZE];
        float[] out = new float[BUFFER_SIZE];

        for (int i = 0; i < BUFFER_SIZE; ++i) {
            in[i] = i;
        }

        long time = Long.MAX_VALUE;
        for (int c = 0; c < LOOP_COUNT; ++c) {
            long start = System.nanoTime();

            for (int i = 0; i < BUFFER_SIZE; ++i) {
                bench(in, out, BUFFER_SIZE, i);
            }

            long end = System.nanoTime();

            time = Math.min(end - start, time);

            print(out);
        }

        System.out.println(String.format("%,.2f ms", time / 1000000.0));
    }

    private static void bench(float[] in, float[] out, int n, int i) {
        float x = in[i];

        float total = 0.0f;
        for (int j = 0; j < 100; ++j) {
            total += Math.cos(x * j) * Math.sin(j);
        }
        out[i] = total;
    }

    private static void bench(CLContext context) throws IOException {
        CLQueue queue = context.createDefaultQueue();
        ByteOrder byteOrder = context.getByteOrder();

        Pointer<Float> inPtr =
                Pointer.allocateFloats(BUFFER_SIZE).order(byteOrder);
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            inPtr.set(i, (float) i);
        }

        CLBuffer<Float> in = context.createFloatBuffer(Usage.Input, inPtr);
        CLBuffer<Float> out =
                context.createFloatBuffer(Usage.Output, BUFFER_SIZE);

        String str =
                IOUtils.readTextClose(JavaCLSample.class.getResourceAsStream("bench.cl"));

        CLProgram program = context.createProgram(str).build();

        CLKernel kernel = program.createKernel("bench", in, out, BUFFER_SIZE);

        long time = Long.MAX_VALUE;
        for (int c = 0; c < LOOP_COUNT; ++c) {
            long start = System.nanoTime();

            CLEvent event =
                    kernel.enqueueNDRange(queue, new int[] { BUFFER_SIZE });

            Pointer<Float> result = out.read(queue, event).order(byteOrder);

            long end = System.nanoTime();

            time = Math.min(end - start, time);

            print((float[]) result.getArray());
        }

        System.out.println(String.format("%,.2f ms", time / 1000000.0));
    }

    private static void print(float[] result) {
        StringBuilder sb = new StringBuilder();
        for (float f : result) {
            sb.append(String.format("%.2f,", f));
        }
        // System.out.println(sb);
    }
}
