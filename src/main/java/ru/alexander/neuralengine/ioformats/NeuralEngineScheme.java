package ru.alexander.neuralengine.ioformats;

import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.ProjectIOFormat;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;

public class NeuralEngineScheme implements ProjectIOFormat {

    @Override
    public boolean isRightFormat(String format) {
        return format.equals("nes");
    }

    @Override
    public void load(FileInputStream stream, GpuExecutor executor) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(stream.readAllBytes());
        stream.close();



        buffer.position(50);

        byte[] buff;
        int instructionCount = buffer.getInt();
        for (int i = 0; i < instructionCount; i++) {
            int nameLen = buffer.getInt();
            buff = new byte[nameLen];
            buffer.get(buff, 0, nameLen);
            String instruction = new String(buff);
            if (!executor.hasInstruction(instruction))
                throw new IllegalStateException("Description for instruction " + instruction + " not added!");
        }


        int varsCount = buffer.getInt();
        for (int i = 0; i < varsCount; i++) {
            int nameLen = buffer.getInt();
            buff = new byte[nameLen];
            buffer.get(buff, 0, nameLen);
            executor.addVariable(new String(buff), buffer.getInt(), buffer.getInt());
        }


        int nameLen = buffer.getInt();
        buff = new byte[nameLen];
        buffer.get(buff, 0, nameLen);
        executor.compile(new String(buff));
    }

    @Override
    public void save(FileOutputStream stream, Data data, GpuExecutor executor) throws IOException {
        AtomicInteger size = new AtomicInteger(62 + data.code.length());
        data.instructions.forEach((name, instruction) -> size.addAndGet(4 + name.length()));
        data.vars.forEach((name, mtx) -> size.addAndGet(12 + name.length()));

        ByteBuffer buffer = ByteBuffer.allocate(size.get());



        InputStream mvnPropFileStream = getClass().getClassLoader()
                .getResourceAsStream("META-INF/maven/ru.alexander/NeuralEngine/pom.properties");
        if (mvnPropFileStream != null) {
            Properties props = new Properties();
            props.load(mvnPropFileStream);
            buffer.put((props.getProperty("version") + " ").getBytes());
        }
        else buffer.put("unknown ".getBytes());

        buffer.position(50);



        buffer.putInt(data.instructions.size());
        data.instructions.forEach((name, instruction) -> {
            buffer.putInt(name.length());
            buffer.put(name.getBytes());
        });

        buffer.putInt(data.vars.size());
        data.vars.forEach((name, mtx) -> {
            buffer.putInt(name.length());
            buffer.put(name.getBytes());
            buffer.putInt(mtx.width());
            buffer.putInt(mtx.height());
        });


        buffer.putInt(data.code.length());
        buffer.put(data.code.getBytes());

        stream.write(buffer.array());
        stream.close();
    }
}
