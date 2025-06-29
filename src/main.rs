use micrograd_rs::nn::Module;
use micrograd_rs::{MLP, MakeMoonDataset, Value};
use ndarray::{Array, Array2};
use std::ops::{Div, Mul};

fn main() {
    demo_mlp();
}

fn demo_mlp() {
    let mlp = MLP::new(2, &[16, 16, 1]); // 2 ->16 ->16 ->1

    let n: usize = 500;
    let dataset = MakeMoonDataset::new(n);

    let total_epoch = 50;
    for epoch in 0..total_epoch {
        let mut correct = 0.0;
        let mut loss = Value::new_with_name(0.0, "loss".to_string());

        for idx in 0..dataset.len() {
            let data_n_label = dataset.get(idx);
            let (data, label) = data_n_label;
            if label != 1.0 && label != -1.0 {
                panic!("label must be -1.0 or 1.0,lable {}", label);
            }

            let output = mlp.forward(&data);
            // if idx % 100 == 0 {
            //     println!("{}", output[0].data())
            // }

            let current_loss = cal_loss(&output, &Value::new(label));

            loss = &loss + &current_loss;
            if (output[0].data() > 0.0 && label == 1.0) || (output[0].data() < 0.0 && label == -1.0)
            {
                correct += 1.0
            }
        }
        let acc = correct / dataset.len() as f64;
        loss = &loss / &(Value::new(dataset.len() as f64));
        let alpha = 0.0001;

        for p in mlp.parameters().iter() {
            loss = &loss + &(&Value::new(alpha) * &(&p.value() * &p.value()))
        }

        mlp.zero_grad();
        loss.backward();
        // # update (sgd)
        let learning_rate = (1.0 - 0.9 * (epoch as f64) / total_epoch as f64) * 0.05;
        for p in mlp.parameters().iter() {
            p.value().add_data(-1.0 * learning_rate * p.value().grad());
        }

        if epoch % 1 == 0 {
            println!(
                "epoch: {}, loss: {}, acc: {}, mlp.parameters[10].value:{},mlp.parameters[10].grad:{},mlp.parameters.len:{}",
                epoch,
                loss.data(),
                acc,
                // learning_rate, learning_rate:{},
                mlp.parameters()[20].value().data(),
                mlp.parameters()[20].value().grad(),
                mlp.parameters().len(),
            );
        }
        if acc > 0.9 {
            break;
        }
    }

    plot_pred_result("moon_dataset_pred.png", &dataset, &mlp)
}

fn plot_pred_result(pic_name: &str, dataset: &MakeMoonDataset, mlp: &MLP) {
    use plotters::prelude::*;

    // 定义绘图区域
    let root_area = BitMapBackend::new(pic_name, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    // 构建坐标轴
    let mut chart = ChartBuilder::on(&root_area)
        .caption("Moon Dataset with Decision Boundary", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-2.0..2.5, -1.0..2.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // 生成网格点
    let point_length = 100;
    let x_vals = Array::linspace(-2.0, 2.5, point_length);
    let y_vals = Array::linspace(-1.0, 2.0, point_length);
    // let mut xx = Array2::<f64>::zeros((x_vals.len(), y_vals.len()));
    // let mut yy = Array2::<f64>::zeros((x_vals.len(), y_vals.len()));

    let mut pred_res_points = Vec::new();
    for i in 0..x_vals.len() {
        for j in 0..y_vals.len() {
            let x = x_vals[i];
            let y = y_vals[j];
            // 模拟分类器预测函数（替换为实际模型逻辑）
            let score = x; // 示例逻辑：圆形边界
            if mlp.forward(&vec![Value::new(x), Value::new(y)])[0].data() > 0.0 {
                pred_res_points.push((x, y));
            }
        }
    }

    // // 预测网格点的分类结果（模拟）
    // let mut z = vec![];
    // for i in 0..x_vals.len() {
    //     for j in 0..y_vals.len() {
    //         let x = xx[(i, j)];
    //         let y = yy[(i, j)];
    //
    //         // 模拟分类器预测函数（替换为实际模型逻辑）
    //         let score = x; // 示例逻辑：圆形边界
    //         if (y - x * x).abs() < 0.01 {
    //             z.push((x, y));
    //             continue;
    //         }
    //     }
    // }

    // let pred_res_points: Vec<(f64, f64)> = (0..dataset.len())
    //     .filter(|&i| {
    //         // let (x, _) = dataset.get(i);
    //         // let pred = mlp.forward(&x);
    //         // return pred[0].data() > 0.0;
    //         return true;
    //     })
    //     .map(|i| {
    //         let (x, _) = dataset.get(i);
    //         (x[0].data(), x[1].data())
    //     })
    //     .collect();

    // 绘制等高线
    // 绘制决策边界区域
    // let _ = chart.draw_series(LineSeries::new(
    //     pred_res_points.iter().map(|&point| point),
    //     &CYAN,
    // ));
    let _ = chart.draw_series(LineSeries::new(
        pred_res_points.iter().map(|&point| point),
        &CYAN,
    ));

    chart
        .draw_series(
            pred_res_points
                .iter()
                .map(|&point| TriangleMarker::new(point, 10, &CYAN)),
        )
        .unwrap();

    // 绘制散点图
    let out_circ_points: Vec<(f64, f64)> = (0..dataset.len())
        .filter(|&i| dataset.label[i] == 1.0)
        .map(|i| {
            let (x, _) = dataset.get(i);
            (x[0].data(), x[1].data())
        })
        .collect();

    let in_circ_points: Vec<(f64, f64)> = (0..dataset.len())
        .filter(|&i| dataset.label[i] == -1.0)
        .map(|i| {
            let (x, _) = dataset.get(i);
            (x[0].data(), x[1].data())
        })
        .collect();

    chart
        .draw_series(
            out_circ_points
                .iter()
                .map(|&point| TriangleMarker::new(point, 5, &RED)),
        )
        .unwrap()
        .label("target 1.0");

    chart
        .draw_series(
            in_circ_points
                .iter()
                .map(|&point| TriangleMarker::new(point, 5, &BLUE)),
        )
        .unwrap()
        .label("target -1.0");

    root_area.present().expect("Failed to save image");
}

#[warn(dead_code)]
fn test_grad() {
    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 = Value::new_with_name(3.0, "node2".to_string());
    let node3 = &node1 * &node2;

    node3.backward();

    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);
    // ValueInner { data: 3.0, grad: 1.0, _op: "*", name: "" }
    // ValueInner { data: 3.0, grad: 1.0, _op: "", name: "node2" }
    // ValueInner { data: 1.0, grad: 3.0, _op: "", name: "node1" }

    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 = Value::new_with_name(1.0, "node2".to_string());
    let mut node3 = &node1 + &node2;

    node3.backward();

    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);
    // Value { data: 1.0, grad: 1.0, _op: "", name: "node1" }
    // Value { data: 1.0, grad: 1.0, _op: "", name: "node2" }
    // Value { data: 2.0, grad: 1.0, _op: "+", name: "" }

    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 = Value::new_with_name(3.0, "node2".to_string());

    node3 = node1.pow_i(&node2);
    node3.backward();
    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);
    // Value { data: 1.0, grad: 3.0, _op: "", name: "node1" }
    // Value { data: 3.0, grad: 0.0, _op: "", name: "node2" }
    // Value { data: 1.0, grad: 1.0, _op: "pow", name: "" }

    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 = Value::new_with_name(3.0, "node2".to_string());
    let node3 = Value::new_with_name(-1.0, "node3".to_string());

    let node4 = node2.pow_i(&node3);
    let node5 = node1.mul(&node4);
    node5.backward();
    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);

    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 = Value::new_with_name(3.0, "node2".to_string());

    let node3 = node1.div(&node2);
    node3.backward();
    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);
    // =

    // let value2 =3.0+ &node1 ;  要实现数字在左边，要么就用宏快速的为各种整数类型来实现，要么就只能数字在右边
}

/// svm "max-margin" loss ， its difficult ,because label is 1or0 not 1or-1
fn cal_loss(out: &Vec<Value>, y: &Value) -> Value {
    if out.len() != 1 {
        panic!("out.len()!=1");
    }
    // let out = &Value::new(1.0) - &Value::new(2.0) * &out[0])*&(p1 - &Value::new(0.5))
    let loss = &Value::new(1.0) - &(&out[0] * y);
    loss.relu()
}
