import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu'; // También probar con '@tensorflow/tfjs-backend-webgl'

export async function benchmarkTFJS(
    X_raw: Float32Array, y_raw: Float32Array, 
    m: number, n: number, alpha: number, epochs: number
) {
    await tf.setBackend('webgpu');
    await tf.ready();

    const tUpload = performance.now();
    const X = tf.tensor2d(X_raw, [m, n]);
    const y = tf.tensor2d(y_raw, [m, 1]);
    let theta = tf.zeros([n, 1]);
    await theta.data(); // Sincronizar subida
    const hostToDeviceMs = performance.now() - tUpload;

    const tTrain = performance.now();
    for (let i = 0; i < epochs; i++) {
        const nextTheta = tf.tidy(() => {
            const pred = tf.matMul(X, theta);
            const error = tf.sub(pred, y);
            const grad = tf.div(tf.matMul(X, error, true, false), m);
            return tf.sub(theta, tf.mul(alpha, grad));
        });
        theta.dispose();
        theta = nextTheta;
    }
    const finalTheta = await theta.data(); // Fuerza ejecución del grafo lazy
    const trainingLoopMs = performance.now() - tTrain;

    X.dispose(); y.dispose(); theta.dispose();
    return { hostToDeviceMs, trainingLoopMs, msPerEpoch: trainingLoopMs / epochs, finalTheta };
}