use crate::{Prev, Value};
use rand::Rng;
use std::ops::Mul;

pub trait Module {
    // fn forward(&self, x: &Value) -> Value;

    fn zero_grad(&self) {
        for p in self.parameters().iter() {
            p.0.borrow_mut().grad = 0.0;
        }
    }

    fn parameters(&self) -> Vec<Prev>;

    fn forward(&self, x: &Vec<Value>) -> Vec<Value>;
}

struct Neuron {
    w: Vec<Value>,
    b: Value,
    non_lin: bool,
}

impl Neuron {
    fn new(n_in: usize, non_lin: bool) -> Neuron {
        let mut rng = rand::rng();

        let w = (0..n_in)
            .map(|_| Value::new(rng.random_range(-1.0..=1.0)))
            .collect();
        Neuron {
            w,
            b: Value::new(rand::random::<f64>()),
            non_lin,
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Prev> {
        let mut vec = vec![];
        for one in self.w.iter() {
            vec.push(Prev(one.0.clone()));
        }
        vec.push(Prev(self.b.0.clone()));
        vec
    }

    fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        if x.len() != self.w.len() {
            panic!("x.len() != self.w.len()");
        }
        let mut out = self.w[0].mul(&x[0]);
        for i in 1..x.len() {
            out = &out + &self.w[i].mul(&x[i]);
        }
        out = &out + &self.b;
        if self.non_lin {
            out = out.relu();
        }
        vec![out]
    }
}

struct Layer {
    ns: Vec<Neuron>,
}
impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize, non_lin: bool) -> Layer {
        let ns = (0..n_outputs)
            .map(|_| Neuron::new(n_inputs, non_lin))
            .collect::<Vec<_>>();
        Layer { ns }
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Prev> {
        let mut vec = vec![];
        for one in self.ns.iter() {
            let mut ps = one.parameters();
            vec.append(&mut ps);
        }
        vec
    }

    fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        let mut out = vec![];
        for one in self.ns.iter() {
            let mut neuron_res = one.forward(x);
            out.append(&mut neuron_res)
        }
        out
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(n_inputs: usize, n_outputs: &[usize]) -> MLP {
        let mut x = n_outputs.to_vec();
        x.insert(0, n_inputs);
        let mut layers = vec![];
        for i in 0..x.len() - 1 {
            let (in_num, out_num) = (x[i], x[i + 1]);
            let mut non_line = true;
            if i == x.len() - 2 {
                // last layer
                non_line = false;
            }
            layers.push(Layer::new(in_num, out_num, non_line));
        }
        MLP { layers }
    }
}
impl Module for MLP {
    fn parameters(&self) -> Vec<Prev> {
        let mut parameters = vec![];
        for layer in self.layers.iter() {
            parameters.append(&mut layer.parameters());
        }
        /* `Vec<Prev>` value */
        parameters
    }

    fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        let mut y = vec![];
        for one in x {
            y.push((*one).clone())
        }
        for layer in self.layers.iter() {
            y = layer.forward(&y);
        }
        y
    }
}
