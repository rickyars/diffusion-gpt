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
        const sessionOptions = {
            executionProviders: [{
                name: 'webgpu',
                device_id: 0,
                preferredOutputLocation: 'cpu',
            }],
            graphOptimizationLevel: 'all',
            enableMemoryPattern: false,
            executionMode: 'sequential'
        };

        try {
            this.session = await ort.InferenceSession.create(this.modelBytes, sessionOptions);
        } catch (e) {
            console.log('WebGPU failed, falling back to WASM:', e.message);
            this.session = await ort.InferenceSession.create(this.modelBytes, {
                executionProviders: ['wasm']
            });
        }
    }

    /**
     * Pre-allocate reusable buffers for a given context length
     */
    prepareBuffers(contextLength) {
        this._tokenBuffer = new Int32Array(contextLength);
        this._probBuffer = new Float64Array(this.vocabSize);
        this._inputIdsBigInt = new BigInt64Array(contextLength);
        this._sigmaBuffer = new Float32Array(1);
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
        this._sigmaBuffer[0] = sigma;
        
        // Convert current token buffer to BigInt for the model input
        for (let i = 0; i < this._tokenBuffer.length; i++) {
            this._inputIdsBigInt[i] = BigInt(this._tokenBuffer[i]);
        }
        
        const inputIdsTensor = new ort.Tensor('int64', this._inputIdsBigInt, [1, this._tokenBuffer.length]);
        const sigmaTensor = new ort.Tensor('float32', this._sigmaBuffer, [1, 1]);

        try {
            const results = await this.session.run({
                'input_ids': inputIdsTensor,
                'sigma': sigmaTensor
            });

            const flatLogits = await results.logits.getData();

            // NOW it is safe to kill the tensors because the data is in CPU memory
            for (const key in results) {
                if (results[key].dispose) results[key].dispose();
            }

            return flatLogits;
        } finally {
            inputIdsTensor.dispose();
            sigmaTensor.dispose();
        }
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

            // 1. Find Max for numerical stability
            let maxL = -Infinity;
            for (let j = 0; j < vocabSize; j++) {
                if (flatLogits[offset + j] > maxL) maxL = flatLogits[offset + j];
            }

            // 2. Compute scoreSum
            let scoreSum = 0;
            for (let j = 0; j < vocabSize; j++) {
                // Using Math.exp(logits - maxL) prevents overflow
                scoreSum += Math.exp(flatLogits[offset + j] - maxL);
            }

            const correction = correctionFactor * scoreSum;
            let probSum = 0;

            // 3. Compute Probabilities
            for (let j = 0; j < vocabSize; j++) {
                const score = Math.exp(flatLogits[offset + j] - maxL);
                const stagScore = correction + score * invExpDelta;
                const transProb = (j === currentToken) ? transDiagProb : transBaseProb;
                
                // Ensure we don't get negative probs from floating point jitter
                const prob = Math.max(0, stagScore * transProb);
                probBuffer[j] = prob;
                probSum += prob;
            }

            // 4. Sample
            const rand = Math.random() * probSum;
            let cumsum = 0;
            let sampled = currentToken; // Default to current if sampling fails

            for (let j = 0; j < vocabSize; j++) {
                cumsum += probBuffer[j];
                if (rand <= cumsum) {
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

            // 100ms "breather" becuase otherise the animation would be too fast
            await new Promise(resolve => setTimeout(resolve, 100));

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
