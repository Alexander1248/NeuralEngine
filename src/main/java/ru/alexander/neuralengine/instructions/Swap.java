package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import jcuda.Sizeof;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

import static jcuda.driver.JCudaDriver.cuMemcpyDtoD;

public class Swap extends Instruction {
    public Swap(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "swap";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);

        cuMemcpyDtoD(out.pointer(), in.pointer(), (long) Sizeof.FLOAT * in.width() * in.height());
        switch (args[4]) {
            case "columns" -> startGPUTask("mtxOperations.swapColumns",
                    in.height(), 1, 1,
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(new int[] { in.height() }),
                    Pointer.to(new int[] { Integer.parseInt(args[2]) }),
                    Pointer.to(new int[] { Integer.parseInt(args[3]) }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
            case "rows" -> startGPUTask("mtxOperations.swapRows",
                    in.width(), 1, 1,
                    Pointer.to(new int[] { in.width() }),
                    Pointer.to(new int[] { Integer.parseInt(args[2]) }),
                    Pointer.to(new int[] { Integer.parseInt(args[3]) }),
                    Pointer.to(in.pointer()),
                    Pointer.to(out.pointer())
            );
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 5)
            throw new IllegalStateException("Instruction format error!");

        if (!args[4].equals("columns") && !args[4].equals("rows"))
            throw new IllegalStateException("Wrong border type!");

        try {
            Integer.parseInt(args[2]);
            Integer.parseInt(args[3]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], var.width(), var.height()))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], var.width(), var.height());
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0 };
    }

    @Override
    public String documentation() {
        return """
                flip <out> <in> <index 1> <index 2> <direction>""";
    }
}
