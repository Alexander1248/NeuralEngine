package ru.alexander.neuralengine.ioformats;

import ru.alexander.neuralengine.executor.GpuExecutor;

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


        byte[] buff = new byte[50];

        buffer.get(buff, 0, 50);
        String msg = new String(buff);

        String version = msg.substring(0, msg.indexOf(" "));
        int pos = version.indexOf("-");
        if (pos != -1) version = version.substring(0, pos);

        if (!version.equals("unknown")) {
            InputStream mvnPropFileStream = getClass().getClassLoader()
                    .getResourceAsStream("engine.properties");
            if (mvnPropFileStream != null) {
                Properties props = new Properties();
                props.load(mvnPropFileStream);

                String mvnVersion = props.getProperty("version");
                pos = mvnVersion.indexOf("-");
                if (pos != -1) mvnVersion = mvnVersion.substring(0, pos);

                pos = version.indexOf(".");
                int major = Integer.parseInt(version.substring(0, pos));
                int minor = Integer.parseInt(version.substring(pos + 1).replace(".", ""));

                pos = mvnVersion.indexOf(".");
                int mvnMajor = Integer.parseInt(mvnVersion.substring(0, pos));
                int mvnMinor = Integer.parseInt(mvnVersion.substring(pos + 1).replace(".", ""));

                if (major != mvnMajor) throw new IllegalStateException("Wrong major version!");
                if (minor > mvnMinor)
                    throw new IllegalStateException("File is too new for this version!");
            }
        }



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
                .getResourceAsStream("engine.properties");
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
