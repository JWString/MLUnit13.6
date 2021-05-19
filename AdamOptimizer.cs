using System;

namespace NNetService.NNet.Training.Modifiers
{
    public class AdamOptimizer<T> : TrainingModifier<T>
    {
        readonly T b1;
        readonly T b2;
        double b1_t;
        double b2_t;

        private T CastT(dynamic arg)
        {
            return (T)arg;
        }

        private T Sqrt(dynamic arg)
        {
            return typeof(T) == typeof(float) ? MathF.Sqrt(arg) : Math.Sqrt(arg);
        }

        private struct ExtendedState
        {
            public T m;
            public T v;
        }

        public AdamOptimizer(float b1 = 0.9F, float b2 = 0.999F)
        {
            this.b1 = CastT(b1);
            this.b2 = CastT(b2);
        }

        public override void OnTrainingStart(ITrainingContext<T> context, ITrainingState<T> state)
        {
            base.OnTrainingStart(context, state);

            b1_t = (dynamic)b1;
            b2_t = (dynamic)b2;

            foreach(var nd in state.neuronData)
            {
                foreach(var wd in nd.weightData)
                {
                    wd.extendedState = new ExtendedState() { m = CastT(0), v = CastT(0) };
                }
            }
        }

        public override void OnWeightUpdate(ITrainingContext<T> context, ITrainingState<T> state, INeuronTrainingState<T> nstate, IWeightTrainingState<T> wstate)
        {
            base.OnWeightUpdate(context, state, nstate, wstate);

            T m = (b1 * wstate.extendedState.m) + ((1 - (dynamic)b1) * wstate.dwSum);
            T v = (b2 * wstate.extendedState.v) + ((1 - (dynamic)b2) * wstate.dwSum * wstate.dwSum);
            T m_hat = CastT(m / (1 - (dynamic)b1_t));
            T v_hat = CastT(v / (1 - (dynamic)b2_t));
            wstate.delta = (context.learningRate / (Sqrt(v_hat) + (dynamic)Single.Epsilon)) * m_hat;
            wstate.extendedState.m = m;
            wstate.extendedState.v = v;
        }

        public override void OnEpochEnd(ITrainingContext<T> context, ITrainingState<T> state)
        {
            base.OnEpochEnd(context, state);

            b1_t *= (dynamic)b1;
            b2_t *= (dynamic)b2;
        }
    }
}
