use crate::{Prev, Value};
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

impl Value {
    pub fn relu(&self) -> Value {
        let out = Value::new_with_name(self.data().max(0.0), "ReLU".to_string());
        out.0.borrow_mut()._prev.insert(Prev(self.0.clone()));

        let self_weak = Rc::downgrade(&self.0);
        let out_weak = Rc::downgrade(&out.0);
        out.0.borrow_mut()._backward = Some(Box::new(move || {
            if let (Some(self_rc), Some(out_rc)) = (self_weak.upgrade(), out_weak.upgrade()) {
                let out_grad = out_rc.borrow().grad;
                if out_grad >= 0.0 {
                    self_rc.borrow_mut().grad += out_grad;
                }
            }
        }));
        out
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        let out = Value::new(self.data() + rhs.data());
        out.0.borrow_mut()._op = "+".to_string();
        out.0.borrow_mut()._prev.insert(Prev(self.0.clone()));
        out.0.borrow_mut()._prev.insert(Prev(rhs.0.clone()));
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
        out.0.borrow_mut()._prev.insert(Prev(self.0.clone()));
        out.0.borrow_mut()._prev.insert(Prev(rhs.0.clone()));
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

impl<T: Into<f64>> Sub<T> for &Value {
    type Output = Value;
    fn sub(self, rhs: T) -> Value {
        let rhs_value = rhs.into().into();
        self.sub(&rhs_value)
    }
}

impl Mul for &Value {
    type Output = Value;
    //todo self 和rhs是同一个valueInner的话会panic
    fn mul(self, rhs: &Value) -> Self::Output {
        let out = Value::new(self.data() * rhs.data());
        out.0.borrow_mut()._op = "*".to_string();
        out.0.borrow_mut()._prev.insert(Prev(self.0.clone()));
        out.0.borrow_mut()._prev.insert(Prev(rhs.0.clone()));
        let self_weak = Rc::downgrade(&self.0);
        let rhs_weak = Rc::downgrade(&rhs.0);
        let out_weak = Rc::downgrade(&out.0);
        out.0.borrow_mut()._backward = Some(Box::new(move || {
            if let (Some(self_rc), Some(other_rc), Some(out_rc)) =
                (self_weak.upgrade(), rhs_weak.upgrade(), out_weak.upgrade())
            {
                let other_data = other_rc.borrow().data;
                let self_data = self_rc.borrow().data;
                let out_grad = out_rc.borrow().grad;
                self_rc.borrow_mut().grad += out_grad * other_data;
                other_rc.borrow_mut().grad += out_grad * self_data;
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
        let value2 = rhs.pow_i(&Value::new_with_name(-1.0, "pow -1.0".to_string()));
        self.mul(&value2)
    }
}

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
        out.0.borrow_mut()._prev.insert(Prev(self.0.clone()));
        out.0.borrow_mut()._prev.insert(Prev(rhs.0.clone()));

        let self_weak = Rc::downgrade(&self.0);
        let rhs_weak = Rc::downgrade(&rhs.0);
        let out_weak = Rc::downgrade(&out.0);
        out.0.borrow_mut()._backward = Some(Box::new(move || {
            if let (Some(self_rc), Some(other_rc), Some(out_rc)) =
                (self_weak.upgrade(), rhs_weak.upgrade(), out_weak.upgrade())
            {
                let out_grad = out_rc.borrow().grad;
                let other_data = other_rc.borrow().data;
                let mut sb = self_rc.borrow_mut();
                let mut ob = other_rc.borrow_mut();
                sb.grad += ob.data * sb.data.powf(other_data - 1.0) * out_grad;
                ob.grad += sb.data.powf(other_data) * sb.data.ln() * out_grad;
            }
        }));
        out
    }

    pub fn pow<T: Into<f64>>(&self, rhs: T) -> Value {
        let x = rhs.into().into();
        self.pow_i(&x)
    }
}
