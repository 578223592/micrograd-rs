use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Add;
use std::rc::{Rc, Weak};

#[derive()]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner::new(data))))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
}
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.borrow();
        f.debug_struct("Value")
            .field("data", &inner.data)
            .field("grad", &inner.grad)
            .field("_op", &inner._op)
            .finish()
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        let out = Value::new(self.data() + rhs.data());
        out.0
            .borrow_mut()
            ._prev
            .insert(Prev(Rc::downgrade(&self.0)));
        out.0.borrow_mut()._prev.insert(Prev(Rc::downgrade(&rhs.0)));

        let self_weak = Rc::downgrade(&self.0);
        let rhs_weak = Rc::downgrade(&self.0);
        let out_weak = Rc::downgrade(&out.0);
        out.0.borrow_mut()._backward = Some(Box::new(move || {
            if let (Some(self_rc), Some(other_rc), Some(out_rc)) =
                (self_weak.upgrade(), rhs_weak.upgrade(), out_weak.upgrade())
            {
                let out_grad = out_rc.borrow().grad;
                self_rc.borrow_mut().grad += out_grad;
                other_rc.borrow_mut().grad += out_grad;
            }
        }));
        out
    }
}

// 新增：实现 Node 与 f64 的加法
impl Add<f64> for Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Value {
        let value = Value::new(rhs);
        self.add(value)
    }
}

// 新增：实现 f64 + Node
impl Add<Value> for f64 {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        rhs.add(self)
    }
}

struct ValueInner {
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
        };
        inner
    }
}

// type Prev = Rc<RefCell<ValueInner>>;
pub struct Prev(Weak<RefCell<ValueInner>>);

impl Hash for Prev {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // 基于 Weak 指针本身的地址做哈希
        let ptr = self.0.as_ptr() as *const ();
        ptr.hash(state);
    }
}

impl PartialEq for Prev {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0.upgrade().unwrap(), &other.0.upgrade().unwrap())
    }
}

impl Eq for Prev {}
