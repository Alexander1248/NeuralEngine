package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class Conv extends Instruction {
    public Conv(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "conv";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);
        Matrix mtx = getVariable(args[2]);

        switch (args[3]) {
            case "empty" -> startGPUTask("mtxOperations.matrixConvEmptyBorder",
                    in.width(), in.height(), 1,
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(new int[] { in.height() }),
                    Pointer.to(new int[] { mtx.width() }),
                    Pointer.to(new int[] { mtx.height() }),
                    Pointer.to(in.pointer()),
                    Pointer.to(mtx.pointer()),
                    Pointer.to(out.pointer())
            );
            case "extend" -> startGPUTask("mtxOperations.matrixConvExtendBorder",
                    in.width(), in.height(), 1,
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(new int[] { in.height() }),
                    Pointer.to(new int[] { mtx.width() }),
                    Pointer.to(new int[] { mtx.height() }),
                    Pointer.to(in.pointer()),
                    Pointer.to(mtx.pointer()),
                    Pointer.to(out.pointer())
            );
            case "repeat" -> startGPUTask("mtxOperations.matrixConvRepeatBorder",
                    in.width(), in.height(), 1,
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(new int[] { in.height() }),
                    Pointer.to(new int[] { mtx.width() }),
                    Pointer.to(new int[] { mtx.height() }),
                    Pointer.to(in.pointer()),
                    Pointer.to(mtx.pointer()),
                    Pointer.to(out.pointer())
            );
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 4)
            throw new IllegalStateException("Instruction format error!");

        if (!args[3].equals("empty") && !args[3].equals("extend") && !args[3].equals("repeat"))
            throw new IllegalStateException("Wrong border type!");

        Matrix var1 = getVariable(args[1]);
        Matrix var2 = getVariable(args[2]);

        if (var1 == null || var2 == null)
            throw new IllegalStateException("Variable not exists!");


        addVariable(args[0], var1.width(), var1.height());
    }
}
