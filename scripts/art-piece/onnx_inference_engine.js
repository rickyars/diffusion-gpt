/**
 * ONNX Runtime Web Inference Engine for Discrete Diffusion
 * Optimized for mobile memory stability - uses pre-allocated flat buffers
 */

class DiffusionInferenceEngine {
    constructor(modelBytes, vocab, config) {
        this.modelBytes = modelBytes;
        this.vocab = vocab;
        this.config = config;
        this.session = null;
        this.itos = vocab.itos;
        this.stoi = vocab.stoi;
        this.vocabSize = vocab.vocab_size;

        // Noise schedule parameters
        this.sigmaMin = 1e-4;
        this.sigmaMax = 20.0;

        // Pre-allocated buffers (initialized in prepareBuffers)
        this._tokenBuffer = null;
        this._probBuffer = null;
        this._inputIdsBigInt = null;
    }

    async init() {
        ort.env.wasm.simd = true;

        if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
            ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        } else {
            ort.env.wasm.numThreads = 1;
        }

        if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
            ort.env.wasm.proxy = false;
        }

        await this._createSession();
    }

    async _createSession() {
        try {
            this.session = await ort.InferenceSession.create(this.modelBytes, {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all'
            });
            this.usingWebGPU = true;
        } catch (e) {
            this.usingWebGPU = false;
            try {
                this.session = await ort.InferenceSession.create(this.modelBytes, {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                });
            } catch (e2) {
                throw new Error('Failed to create ONNX session with any provider');
            }
        }
    }

    /**
     * Pre-allocate reusable buffers for a given context length
     */
    prepareBuffers(contextLength) {
        this._tokenBuffer = new Int32Array(contextLength);
        this._probBuffer = new Float64Array(this.vocabSize);
        this._inputIdsBigInt = new BigInt64Array(contextLength);
    }

    /**
     * Geometric noise schedule (inlined for performance)
     */
    getSigma(t) {
        const logSigmaMin = Math.log(this.sigmaMin);
        const logSigmaMax = Math.log(this.sigmaMax);
        return Math.exp((1 - t) * logSigmaMin + t * logSigmaMax);
    }

    /**
     * Run model inference with explicit tensor lifecycle management
     */
    async runModel(sigma) {
        // Reuse pre-allocated BigInt64Array, just update values
        for (let i = 0; i < this._tokenBuffer.length; i++) {
            this._inputIdsBigInt[i] = BigInt(this._tokenBuffer[i]);
        }

        const inputIdsTensor = new ort.Tensor(
            'int64',
            this._inputIdsBigInt,
            [1, this._tokenBuffer.length]
        );

        const sigmaTensor = new ort.Tensor(
            'float32',
            new Float32Array([sigma]),
            [1, 1]
        );

        let flatLogits;
        try {
            const results = await this.session.run({
                'input_ids': inputIdsTensor,
                'sigma': sigmaTensor
            });
            flatLogits = results.logits.data;

            // Explicit disposal of output tensor
            if (results.logits.dispose) {
                results.logits.dispose();
            }
        } finally {
            // Always dispose input tensors
            inputIdsTensor.dispose();
            sigmaTensor.dispose();
        }

        return flatLogits;
    }

    /**
     * Sample next tokens in-place using pre-allocated buffers
     * Consolidated to 2 passes through vocab (down from 4)
     */
    sampleInPlace(flatLogits, deltaSigma) {
        const vocabSize = this.vocabSize;
        const seqLen = this._tokenBuffer.length;
        const probBuffer = this._probBuffer;

        const expDelta = Math.exp(-deltaSigma);
        const invExpDelta = 1.0 / expDelta;
        const transBaseProb = (1 - expDelta) / vocabSize;
        const transDiagProb = 1 - (vocabSize - 1) * transBaseProb;
        const correctionFactor = (expDelta - 1) / (vocabSize * expDelta);

        for (let i = 0; i < seqLen; i++) {
            const offset = i * vocabSize;
            const currentToken = this._tokenBuffer[i];

            // Pass 1: Find max, compute scores, probabilities, and sum in one loop
            let maxL = flatLogits[offset];
            for (let j = 1; j < vocabSize; j++) {
                if (flatLogits[offset + j] > maxL) maxL = flatLogits[offset + j];
            }

            let scoreSum = 0;
            let probSum = 0;

            for (let j = 0; j < vocabSize; j++) {
                const score = Math.exp(flatLogits[offset + j] - maxL);
                scoreSum += score;
            }

            const correction = correctionFactor * scoreSum;

            for (let j = 0; j < vocabSize; j++) {
                const score = Math.exp(flatLogits[offset + j] - maxL);
                const stagScore = correction + score * invExpDelta;
                const transProb = (j === currentToken) ? transDiagProb : transBaseProb;
                const prob = stagScore * transProb;
                probBuffer[j] = prob;
                probSum += prob;
            }

            // Pass 2: Sample from normalized distribution
            const rand = Math.random() * probSum;
            let cumsum = 0;
            let sampled = vocabSize - 1;

            for (let j = 0; j < vocabSize; j++) {
                cumsum += probBuffer[j];
                if (rand < cumsum) {
                    sampled = j;
                    break;
                }
            }

            this._tokenBuffer[i] = sampled;
        }
    }

    /**
     * Decode token buffer to text
     */
    decodeBuffer() {
        let result = '';
        for (let i = 0; i < this._tokenBuffer.length; i++) {
            result += this.itos[this._tokenBuffer[i]];
        }
        return result;
    }

    /**
     * Yield to main thread - prevents mobile browser timeout
     */
    yieldToMain() {
        return new Promise(resolve => setTimeout(resolve, 0));
    }

    /**
     * Generate with real-time streaming, optimized for mobile
     */
    async* generateStream(contextLength = 256, steps = 128) {
        // Allocate buffers once at start
        this.prepareBuffers(contextLength);

        // Initialize with random tokens
        for (let i = 0; i < contextLength; i++) {
            this._tokenBuffer[i] = Math.floor(Math.random() * this.vocabSize);
        }

        const eps = 1e-5;
        const stepSize = (1 - eps) / steps;

        for (let i = 0; i < steps; i++) {
            const t = 1 - (i + 1) * stepSize;
            const currSigma = this.getSigma(t);

            const deltaSigma = (i < steps - 1)
                ? currSigma - this.getSigma(t - stepSize)
                : currSigma;

            // Run model inference
            const flatLogits = await this.runModel(currSigma);

            // Sample in-place (modifies _tokenBuffer directly)
            this.sampleInPlace(flatLogits, deltaSigma);

            // Yield to main thread every step to prevent timeout
            await this.yieldToMain();

            yield {
                step: i + 1,
                totalSteps: steps,
                text: this.decodeBuffer(),
                sigma: currSigma,
                progress: (i + 1) / (steps + 1)
            };
        }
    }
}
