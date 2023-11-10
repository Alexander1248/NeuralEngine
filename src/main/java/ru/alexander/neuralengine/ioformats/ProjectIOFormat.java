package ru.alexander.neuralengine.ioformats;


import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface ProjectIOFormat {
    boolean isRightFormat(String format);

    void load(FileInputStream stream, GpuExecutor executor) throws IOException;
    void save(FileOutputStream stream, Data data, GpuExecutor executor) throws IOException;

    class Data {
        public Map<String, Instruction> instructions;
        public Map<String, Matrix> vars;
        public String code;
    }
}
