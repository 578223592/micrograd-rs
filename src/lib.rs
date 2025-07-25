mod data;
mod math_cal;
pub mod nn;

use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

// 重新导出结构体，使其对 crate 外部可见
pub use data::MakeMoonDataset;

pub use nn::MLP;

#[derive()]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Value {
    pub(crate) fn clone(&self) -> Value {
        let rc = self.0.clone();
        Value(rc)
    }
}

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner::new(data))))
    }

    pub fn new_with_name(data: f64, name: String) -> Self {
        Value(Rc::new(RefCell::new(ValueInner::new_with_name(data, name))))
    }
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn add_data(&self, add_data: f64) {
        self.0.borrow_mut().data += add_data;
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn backward(&self) {
        // 构建计算图拓扑排序
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo(&mut topo, &mut visited);

        // # go one variable at a time and apply the chain rule to get its gradient
        self.set_grad(1.0);
        topo.reverse();

        for node in topo.iter() {
            // println!("{:?}", &node.borrow());
            if let Some(backward_fn) = &node.borrow()._backward {
                backward_fn();
            }
        }
    }

    fn build_topo(
        &self,
        topo: &mut Vec<Rc<RefCell<ValueInner>>>,
        visited: &mut HashSet<*const RefCell<ValueInner>>,
    ) {
        let ptr = Rc::as_ptr(&self.0) as *const RefCell<ValueInner>;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        for prev in self.0.borrow()._prev.iter() {
            // if let rc = prev.0 {
            Value(prev.0.clone()).build_topo(topo, visited);
            // }
        }
        topo.push(self.0.clone());
    }
}
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.borrow();
        f.debug_struct("Value")
            .field("data", &inner.data)
            .field("grad", &inner.grad)
            .field("_op", &inner._op)
            .field("name", &inner.name)
            .finish()
    }
}

struct ValueInner {
    name: String,
    data: f64,
    grad: f64,
    _backward: Option<Box<dyn Fn()>>,
    _prev: HashSet<Prev>,
    _op: String,
}

impl ValueInner {
    pub fn new(val: f64) -> ValueInner {
        let inner = ValueInner {
            data: val,
            grad: 0.0,
            _backward: None,
            _prev: Default::default(),
            _op: Default::default(),
            name: Default::default(),
        };
        inner
    }
    pub fn new_with_name(val: f64, name: String) -> ValueInner {
        let inner = ValueInner {
            data: val,
            grad: 0.0,
            _backward: None,
            _prev: Default::default(),
            _op: Default::default(),
            name,
        };
        inner
    }
}

impl fmt::Debug for ValueInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self;
        f.debug_struct("ValueInner")
            .field("data", &inner.data)
            .field("grad", &inner.grad)
            .field("_op", &inner._op)
            .field("name", &inner.name)
            .finish()
    }
}
// use RC rather than weak , because we need to keep the ValueInner alive,or the ValueInner will be dropped automatically.
pub struct Prev(Rc<RefCell<ValueInner>>);

impl Prev {
    pub fn value(&self) -> Value {
        Value(self.0.clone())
    }
}

impl Hash for Prev {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash based on the address of the control block
        self.0.as_ptr().hash(state);
    }
}

impl PartialEq for Prev {
    fn eq(&self, other: &Self) -> bool {
        // Safely compares Weak pointers without unwrapping
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Prev {}
