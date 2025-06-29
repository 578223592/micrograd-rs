use crate::Value;
use ndarray::{Array, Axis, Ix1, Ix2, concatenate, stack};
// file from :https://github.com/samsja/rusty-grad , thanks samsja

pub fn make_moon(n_samples: usize) -> [Array<f32, Ix2>; 2] {
    let n_samples_in = n_samples / 2;
    let n_samples_out = n_samples - n_samples_in;

    let pi = std::f32::consts::PI;

    let out_circ_x = Array::linspace(0., pi, n_samples_out).mapv(|x| x.cos());
    let out_circ_y = Array::linspace(0., pi, n_samples_out).mapv(|y| y.sin());

    let in_circ_x = Array::linspace(0., pi, n_samples_in).mapv(|x| 1. - x.cos());
    let in_circ_y = Array::linspace(0., pi, n_samples_in).mapv(|x| 1. - x.sin() - 0.5);

    let out_circ = stack(Axis(0), &[out_circ_x.view(), out_circ_y.view()]).unwrap();
    let in_circ = stack(Axis(0), &[in_circ_x.view(), in_circ_y.view()]).unwrap();

    [out_circ, in_circ]
}

pub struct MakeMoonDataset {
    data: Array<f32, Ix2>,
    pub label: Array<f32, Ix1>,
}

impl MakeMoonDataset {
    pub fn new(n_samples: usize) -> MakeMoonDataset {
        let [out_circ, in_circ] = make_moon(n_samples);

        let data = concatenate(Axis(1), &[in_circ.view(), out_circ.view()]).unwrap();

        let label_out = Array::<f32, Ix1>::zeros(out_circ.shape()[1]);
        let label_in = Array::<f32, Ix1>::ones(in_circ.shape()[1]);

        let label =
            (concatenate(Axis(0), &[label_in.view(), label_out.view()]).unwrap() - 0.5) * 2.0;

        MakeMoonDataset { data, label }
    }

    pub fn len(&self) -> usize {
        self.data.shape()[1]
    }

    pub fn get(&self, idx: usize) -> (Vec<Value>, f64) {
        let data = self.data.column(idx).to_shape((2, 1)).unwrap().mapv(|x| x);
        let (x, _) = data.into_raw_vec_and_offset();
        let x = x
            .iter()
            .map(|x| {
                let x = *x as f64;
                Value::new(x)
            })
            .collect();
        (x, self.label[idx] as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plotters::prelude::{
        BLUE, BitMapBackend, ChartBuilder, IntoDrawingArea, RED, TriangleMarker, WHITE,
    };

    #[test]
    fn make_moon_test() {
        let n: usize = 100;
        let [out_circ, in_circ] = make_moon(2 * n);

        assert_eq!(out_circ.shape(), [2, 100]);
        assert_eq!(in_circ.shape(), [2, 100]);
    }

    #[test]
    fn make_moon_dataset_test() {
        let n: usize = 100;

        let dataset = MakeMoonDataset::new(n);

        let data = dataset.get(0);

        assert_eq!(data.0.len(), 2);
        assert_eq!(data.1, 1.);
    }

    #[test]
    fn moon_plot() {
        let n: usize = 100;

        let dataset = MakeMoonDataset::new(n);

        // 创建绘图区域
        let root_area = BitMapBackend::new("moon_dataset.png", (800, 600)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        // 配置坐标轴
        let mut binding = ChartBuilder::on(&root_area);
        let chart_builder = binding
            .caption("Moon Dataset Visualization", ("sans-serif", 20))
            .x_label_area_size(30)
            .y_label_area_size(30);

        let x_min = -2.0;
        let x_max = 2.5;
        let y_min = -1.0;
        let y_max = 2.0;

        let mut chart = chart_builder
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        // 绘制点
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
            .unwrap();

        chart
            .draw_series(
                in_circ_points
                    .iter()
                    .map(|&point| TriangleMarker::new(point, 5, &BLUE)),
            )
            .unwrap();
    }

    #[test]
    fn moon_plot_with_decision_boundary() {
        use ndarray::{Array, Array2};
        use plotters::prelude::*;

        let n: usize = 100;
        let dataset = MakeMoonDataset::new(n);

        // 定义绘图区域
        let root_area =
            BitMapBackend::new("moon_dataset_with_boundary.png", (800, 600)).into_drawing_area();
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
        let x_vals = Array::linspace(-2.0, 2.5, 100);
        let y_vals = Array::linspace(-1.0, 2.0, 100);
        let mut xx = Array2::<f64>::zeros((x_vals.len(), y_vals.len()));
        let mut yy = Array2::<f64>::zeros((x_vals.len(), y_vals.len()));

        for i in 0..x_vals.len() {
            for j in 0..y_vals.len() {
                xx[(i, j)] = x_vals[i];
                yy[(i, j)] = y_vals[j];
            }
        }

        // 预测网格点的分类结果（模拟）
        let mut z = vec![];
        for i in 0..x_vals.len() {
            for j in 0..y_vals.len() {
                let x = xx[(i, j)];
                let y = yy[(i, j)];

                // 模拟分类器预测函数（替换为实际模型逻辑）
                let score = x; // 示例逻辑：圆形边界
                if (y - x * x).abs() < 0.01 {
                    z.push((x, y));
                    continue;
                }
            }
        }

        // 绘制等高线
        // 绘制决策边界区域
        let _ = chart.draw_series(LineSeries::new(z.iter().map(|&point| point), &CYAN));

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
            .unwrap();

        chart
            .draw_series(
                in_circ_points
                    .iter()
                    .map(|&point| TriangleMarker::new(point, 5, &BLUE)),
            )
            .unwrap();

        root_area.present().expect("Failed to save image");
    }
}
