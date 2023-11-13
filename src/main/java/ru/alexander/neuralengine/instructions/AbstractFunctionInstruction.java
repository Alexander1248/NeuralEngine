package ru.alexander.neuralengine.instructions;

import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;

import java.util.HashMap;
import java.util.Map;

public abstract class AbstractFunctionInstruction extends Instruction {
    private final Map<String[], GpuExecutor> executors = new HashMap<>();

    public AbstractFunctionInstruction(GpuExecutor root) {
        super(root);
    }

    @Override
    public void compute(String... args) {
        loadInstructions(args);
        executors.get(args).compute();
        unloadInstructions(args);
    }

    public void newInstance(GpuExecutor executor, String... args) {
        executors.put(args, executor);
    }

    protected GpuExecutor getFunctionExecutor(String... args) {
        return executors.get(args);
    }

    public abstract void loadInstructions(String... args);
    public abstract void unloadInstructions(String... args);
}
