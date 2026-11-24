export interface BenchmarkMetrics {
    initAndCompileMs: number;
    hostToDeviceMs: number;
    trainingLoopMs: number;
    msPerEpoch: number;
    deviceToHostMs: number;
    finalTheta: Float32Array;
}

export class WebGPULinearRegression {
    private device!: GPUDevice;
    private forwardPipeline!: GPUComputePipeline;
    private backwardPipeline!: GPUComputePipeline;

    async init(forwardShaderCode: string, backwardShaderCode: string): Promise<number> {
        const t0 = performance.now();
        if (!navigator.gpu) throw new Error("WebGPU no soportado en este navegador.");
        
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) throw new Error("No se encontró adaptador GPU.");
        
        this.device = await adapter.requestDevice();

        const forwardModule = this.device.createShaderModule({ code: forwardShaderCode });
        const backwardModule = this.device.createShaderModule({ code: backwardShaderCode });

        this.forwardPipeline = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: forwardModule, entryPoint: "main" }
        });

        this.backwardPipeline = await this.device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: backwardModule, entryPoint: "main" }
        });

        return performance.now() - t0; // Tiempo de inicialización y compilación (warm-up)
    }

    async fit(
        X: Float32Array, 
        y: Float32Array, 
        m: number, 
        n: number, 
        alpha: number, 
        epochs: number
    ): Promise<BenchmarkMetrics> {
        const tUploadStart = performance.now();

        // 1. Creación y carga de Buffers (Host -> Device)
        const uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const uniformData = new ArrayBuffer(16);
        const view = new DataView(uniformData);
        view.setUint32(0, m, true);
        view.setUint32(4, n, true);
        view.setFloat32(8, alpha, true);
        this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const xBuffer = this.createAndUploadBuffer(X, GPUBufferUsage.STORAGE);
        const yBuffer = this.createAndUploadBuffer(y, GPUBufferUsage.STORAGE);
        const errorBuffer = this.device.createBuffer({
            size: m * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        const thetaBuffer = this.device.createBuffer({
            size: n * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        // Inicializar theta en ceros
        this.device.queue.writeBuffer(thetaBuffer, 0, new Float32Array(n));

        // Buffer staging para leer resultados finales en CPU
        const readbackBuffer = this.device.createBuffer({
            size: n * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        await this.device.queue.onSubmittedWorkDone();
        const hostToDeviceMs = performance.now() - tUploadStart;

        // 2. BindGroups
        const forwardBindGroup = this.device.createBindGroup({
            layout: this.forwardPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: xBuffer } },
                { binding: 2, resource: { buffer: yBuffer } },
                { binding: 3, resource: { buffer: thetaBuffer } },
                { binding: 4, resource: { buffer: errorBuffer } },
            ]
        });

        const backwardBindGroup = this.device.createBindGroup({
            layout: this.backwardPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: xBuffer } },
                { binding: 2, resource: { buffer: errorBuffer } },
                { binding: 3, resource: { buffer: thetaBuffer } },
            ]
        });

        // 3. Bucle de entrenamiento (100% residente en GPU)
        const tTrainStart = performance.now();
        const forwardWorkgroups = Math.ceil(m / 256);
        const backwardWorkgroups = n; // Un workgroup por característica

        // Agrupamos comandos en bloques para minimizar overhead de submit en JS
        const BATCH_SIZE = 50; 
        for (let e = 0; e < epochs; e += BATCH_SIZE) {
            const steps = Math.min(BATCH_SIZE, epochs - e);
            const encoder = this.device.createCommandEncoder();

            for (let s = 0; s < steps; s++) {
                const pass1 = encoder.beginComputePass();
                pass1.setPipeline(this.forwardPipeline);
                pass1.setBindGroup(0, forwardBindGroup);
                pass1.dispatchWorkgroups(forwardWorkgroups);
                pass1.end();

                const pass2 = encoder.beginComputePass();
                pass2.setPipeline(this.backwardPipeline);
                pass2.setBindGroup(0, backwardBindGroup);
                pass2.dispatchWorkgroups(backwardWorkgroups);
                pass2.end();
            }
            this.device.queue.submit([encoder.finish()]);
        }

        await this.device.queue.onSubmittedWorkDone();
        const trainingLoopMs = performance.now() - tTrainStart;

        // 4. Descarga de pesos (Device -> Host)
        const tDownloadStart = performance.now();
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(thetaBuffer, 0, readbackBuffer, 0, n * 4);
        this.device.queue.submit([copyEncoder.finish()]);

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const finalTheta = new Float32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        const deviceToHostMs = performance.now() - tDownloadStart;

        // Limpieza de memoria VRAM
        [uniformBuffer, xBuffer, yBuffer, errorBuffer, thetaBuffer, readbackBuffer].forEach(b => b.destroy());

        return {
            initAndCompileMs: 0, // Asignar desde init()
            hostToDeviceMs,
            trainingLoopMs,
            msPerEpoch: trainingLoopMs / epochs,
            deviceToHostMs,
            finalTheta
        };
    }

    private createAndUploadBuffer(data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: usage | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }
}