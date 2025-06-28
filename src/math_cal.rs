use crate::{Prev, Value};
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        let out = Value::new(self.data() + rhs.data());
        out.0.borrow_mut()._op = "+".to_string();
        out.0
            .borrow_mut()
            ._prev
            .insert(Prev(Rc::downgrade(&self.0)));
        out.0.borrow_mut()._prev.insert(Prev(Rc::downgrade(&rhs.0)));

        let self_weak = Rc::downgrade(&self.0);
        let rhs_weak = Rc::downgrade(&rhs.0);
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

//
// // 新增：实现 f64 + Node
// impl Add<&Value> for f64 {
//     type Output = Value;
//
//     fn add(self, rhs: &Value) -> Value {
//         rhs.add(self)
//     }
// }

impl<T: Into<f64>> Add<T> for &Value {
    type Output = Value;
    fn add(self, rhs: T) -> Value {
        let rhs_value = rhs.into().into();
        self.add(&rhs_value)
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: &Value) -> Value {
        let out = Value::new(self.data() - rhs.data());
        out.0.borrow_mut()._op = "-".to_string();
        out.0
            .borrow_mut()
            ._prev
            .insert(Prev(Rc::downgrade(&self.0)));
        out.0.borrow_mut()._prev.insert(Prev(Rc::downgrade(&rhs.0)));

        let self_weak = Rc::downgrade(&self.0);
        let rhs_weak = Rc::downgrade(&rhs.0);
        let out_weak = Rc::downgrade(&out.0);
        out.0.borrow_mut()._backward = Some(Box::new(move || {
            if let (Some(self_rc), Some(other_rc), Some(out_rc)) =
                (self_weak.upgrade(), rhs_weak.upgrade(), out_weak.upgrade())
            {
                let out_grad = out_rc.borrow().grad;
                self_rc.borrow_mut().grad -= out_grad;
                other_rc.borrow_mut().grad -= out_grad;
            }
        }));
        out
    }
}

//
// // 新增：实现 f64 + Node
// impl Add<&Value> for f64 {
//     type Output = Value;
//
//     fn add(self, rhs: &Value) -> Value {
//         rhs.add(self)
//     }
// }

impl<T: Into<f64>> Sub<T> for &Value {
    type Output = Value;
    fn sub(self, rhs: T) -> Value {
        let rhs_value = rhs.into().into();
        self.sub(&rhs_value)
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Self::Output {
        let out = Value::new(self.data() * rhs.data());
        out.0.borrow_mut()._op = "*".to_string();
        out.0
            .borrow_mut()
            ._prev
            .insert(Prev(Rc::downgrade(&self.0)));
        out.0.borrow_mut()._prev.insert(Prev(Rc::downgrade(&rhs.0)));

        let self_weak = Rc::downgrade(&self.0);
        let rhs_weak = Rc::downgrade(&rhs.0);
        let out_weak = Rc::downgrade(&out.0);
        out.0.borrow_mut()._backward = Some(Box::new(move || {
            if let (Some(self_rc), Some(other_rc), Some(out_rc)) =
                (self_weak.upgrade(), rhs_weak.upgrade(), out_weak.upgrade())
            {
                let out_grad = out_rc.borrow().grad;
                self_rc.borrow_mut().grad += out_grad * other_rc.borrow().data;
                other_rc.borrow_mut().grad += out_grad * self_rc.borrow().data;
            }
        }));
        out
    }
}

//
// // 新增：实现 f64 + Node
// impl Add<&Value> for f64 {
//     type Output = Value;
//
//     fn add(self, rhs: &Value) -> Value {
//         rhs.add(self)
//     }
// }

impl<T: Into<f64>> Mul<T> for &Value {
    type Output = Value;
    fn mul(self, rhs: T) -> Value {
        let rhs_value = rhs.into().into();
        self.mul(&rhs_value)
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: &Value) -> Value {
        let value1 = Value::new_with_name(-1.0, "pow -1.0".to_string());
        let value2 = rhs.pow_i(&value1);
        let value = self.mul(&value2);
        value //todo 临时变量会被删除，导致链式法则无法向前传递，因此需要用强指针  2025年06月29日01:15:26
    }
}

//
// // 新增：实现 f64 + Node
// impl Add<&Value> for f64 {
//     type Output = Value;
//
//     fn add(self, rhs: &Value) -> Value {
//         rhs.add(self)
//     }
// }

impl<T: Into<f64>> Div<T> for &Value {
    type Output = Value;
    fn div(self, rhs: T) -> Value {
        let rhs_value = rhs.into().into();
        self.div(&rhs_value)
    }
}

// 实现 From<f64> 用于自动转换
impl From<f64> for Value {
    fn from(data: f64) -> Self {
        Value::new(data)
    }
}

impl Value {
    pub fn pow_i(&self, rhs: &Value) -> Value {
        let out = Value::new(self.data().powf(rhs.data()));
        out.0.borrow_mut()._op = "pow".to_string();
        out.0
            .borrow_mut()
            ._prev
            .insert(Prev(Rc::downgrade(&self.0)));
        out.0.borrow_mut()._prev.insert(Prev(Rc::downgrade(&rhs.0)));

        let self_weak = Rc::downgrade(&self.0);
        let rhs_weak = Rc::downgrade(&rhs.0);
        let out_weak = Rc::downgrade(&out.0);
        out.0.borrow_mut()._backward = Some(Box::new(move || {
            if let (Some(self_rc), Some(other_rc), Some(out_rc)) =
                (self_weak.upgrade(), rhs_weak.upgrade(), out_weak.upgrade())
            {
                let out_grad = out_rc.borrow().grad;
                let mut sb = self_rc.borrow_mut();
                let mut ob = other_rc.borrow_mut();
                sb.grad += ob.data * sb.data.powf(ob.data - 1.0) * out_grad;
                ob.grad += sb.data.powf(ob.data) * sb.data.ln() * out_grad;
            }
        }));
        out
    }

    pub fn pow<T: Into<f64>>(&self, rhs: T) -> Value {
        let x = rhs.into().into();
        self.pow_i(&x)
    }
}
