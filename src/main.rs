use std::io::Write;
use std::{env, io};
use std::{mem, time};

fn is_epsilon(val: f64, compare_val: f64, eps: f64) -> bool {
    if compare_val.abs() <= f64::EPSILON {
        return val.abs() < eps;
    }

    return (val / compare_val).abs() < eps;
}

fn error(v1: &[f64], v2: &[f64]) -> f64 {
    assert_eq!(v1.len(), v2.len());

    let mut summ = 0.0;
    let mut compare = 0.0;
    for i in 0..v1.len() {
        let diff = v1[i] - v2[i];
        summ += diff * diff;
        compare += v1[i].abs().max(v2[i].abs());
    }

    return summ / compare;
}

struct CartesianScheme {
    m_k: f64,
}

struct PolarScheme {
    m_k1: f64,
    m_k2: f64,
}

trait Scheme {
    fn scheme(&self, x: f64, y1: &f64, y2: &f64, y3: &f64) -> f64;
    fn new(koeff: f64, tau: f64, h: f64) -> Self;
}

impl Scheme for CartesianScheme {
    fn scheme(&self, _: f64, y1: &f64, y2: &f64, y3: &f64) -> f64 {
        self.m_k * (y1 - 2.0 * y2 + y3) + y2
    }
    fn new(koeff: f64, tau: f64, h: f64) -> Self {
        CartesianScheme {
            m_k: koeff * tau / h / h,
        }
    }
}

impl Scheme for PolarScheme {
    fn scheme(&self, r: f64, y1: &f64, y2: &f64, y3: &f64) -> f64 {
        self.m_k1 * (y1 - 2.0 * y2 + y3) + self.m_k2 / r * (y3 - y1) + y2
    }
    fn new(koeff: f64, tau: f64, h: f64) -> Self {
        PolarScheme {
            m_k1: koeff * tau / h / h,
            m_k2: koeff * tau / 2.0 / h,
        }
    }
}

struct BoundaryCondition1rd {
    m_u: f64,
}
struct BoundaryCondition2rd {
    m_du: f64,
}
// kU * dU/dx + kdU * U = C
struct BoundaryCondition3rd {
    m_ku: f64,
    m_kdu: f64,
    m_c: f64,
}

trait BoundaryCondition {
    fn calc(&self, h: f64, t_curr: f64, y0: f64, y1: f64) -> f64;
}

impl BoundaryCondition for BoundaryCondition1rd {
    fn calc(&self, _: f64, _: f64, _: f64, _: f64) -> f64 {
        self.m_u
    }
}
impl BoundaryCondition for BoundaryCondition2rd {
    fn calc(&self, h: f64, _: f64, y0: f64, y1: f64) -> f64 {
        (2.0 * h * self.m_du + 4.0 * y0 - y1) / 3.0
    }
}
impl BoundaryCondition for BoundaryCondition3rd {
    fn calc(&self, h: f64, _: f64, y0: f64, y1: f64) -> f64 {
        let divider = 2.0 * h * self.m_ku / self.m_kdu - 3.0;
        if is_epsilon(divider, 3.0, 0.00001) {
            (self.m_c - (y0)) / (h * self.m_ku / self.m_kdu - 1.0) //O(h)
        } else {
            (2.0 * h / self.m_kdu * self.m_c - 4.0 * (y0) + (y1)) / divider //O(h*h)
        }
    }
}

fn explicit_difference_scheme<T: Scheme>(
    y_init: &Vec<f64>,
    koeff: f64,
    a: f64,
    b: f64,
    t: f64,
    bc_a: &dyn BoundaryCondition,
    bc_b: &dyn BoundaryCondition,
    eps: f64,
    max_iter: u32,
) -> Vec<f64> {
    let now = time::Instant::now();
    let size = y_init.len();
    let h = (b - a) / (size - 1) as f64;
    let mut times = (2.0 * koeff * t / h / h + 0.5).round() as u32;
    let mut err: f64;

    assert!(size >= 3, "Need 3 points or more");

    let mut y_prev = y_init.clone();

    for iter in 1..max_iter {
        let mut y = y_init.clone();
        let mut y_tmp = y_init.clone();

        let tau = t / times as f64;
        print!("iter={iter}, h={h}, tau={tau}");
        io::stdout().flush().expect("Uncknown io error");

        let scheme = T::new(koeff, tau, h);
        let mut t_curr = 0.0;

        for _ in 1..times + 1 {
            mem::swap(&mut y, &mut y_tmp);

            t_curr += tau;
            let mut x = a;
            //inner nodes

            for i in 1..(size - 1) {
                x += h;

                let (y1, y2, y3) = unsafe {
                    (
                        y_tmp.get_unchecked(i - 1),
                        y_tmp.get_unchecked(i),
                        y_tmp.get_unchecked(i + 1),
                    )
                };

                y[i] = scheme.scheme(x, y1, y2, y3);
            }

            //boundary conditions
            y[0] = bc_a.calc(h, t_curr, y[1], y[2]);
            y[size - 1] = bc_b.calc(h, t_curr, y[size - 2], y[size - 3]);
        }

        err = error(&y, &y_prev);
        y_prev = mem::take(&mut y);

        println!(", N = {times}, err = {err:>.5e}\n");

        if err < eps {
            break;
        }
        times *= 2;
    }
    let elapsed_time = now.elapsed();
    println!("Elapsed: {} ms\n", elapsed_time.as_millis());

    return y_prev;
}

fn dump_result(y: &Vec<f64>) {
    io::stderr()
        .write_fmt(format_args!("No\t<==>\tcalculated"))
        .expect("Uncknown io error");
    for i in 0..y.len() {
        io::stderr()
            .write_fmt(format_args!("{}\t<==>\t{:>.5}\n", i, y[i]))
            .expect("Uncknown io error");
    }
    io::stdout().flush().expect("Uncknown io error");
}

fn main() {
    let mut test = "test1";
    let mut nodes = 10;
    let mut eps = 0.001;

    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        test = args[1].trim();
        if args.len() > 2 {
            nodes = args[2]
                .trim()
                .parse()
                .expect("Please type a number of nodes");
            if args.len() > 3 {
                eps = args[3].trim().parse().expect("Please type a accuracy");
            }
        }
    }

    println!("Start {} with nodes = {}, eps = {}\n", test, nodes, eps);
    match test {
        "test1" => test1(nodes, eps),
        "test2" => test2(nodes, eps),
        "test3" => test3(nodes, eps),
        _ => println!("No such test"),
    }
}

//TEST dekart  dU/dt = 4 * d2U/dt2 ; U(x,0)=1, U(0,t)=0, U(1,t)=1
fn test1(nodes: usize, eps: f64) {
    let mut y = vec![1.0; nodes];
    let bc_a = BoundaryCondition1rd { m_u: 0.0 };
    let bc_b = BoundaryCondition1rd { m_u: 1.0 };

    y = explicit_difference_scheme::<CartesianScheme>(
        &y, 4.0, 0.0, 1.0, 6.0, &bc_a, &bc_b, eps, 1000,
    );

    dump_result(&y)
}

//TEST dekart  dU/dt = a * d2U/dt2 ; U(x,0)=T0(x), U(0,t)=U(l,t)=0
fn test2(nodes: usize, eps: f64) {
    let mut y = vec![0.0; nodes];
    let l = 100.0;
    let a = 9.0;

    for i in 0..y.len() {
        let x = l / (y.len() - 1) as f64 * i as f64;
        y[i] = if x < l / 2.0 {
            200.0 / l * x
        } else {
            -200.0 / l * x + 200.0
        };
    }

    let t = 1.0;
    let bc_a = BoundaryCondition1rd { m_u: 0.0 };
    let bc_b = BoundaryCondition1rd { m_u: 0.0 };

    y = explicit_difference_scheme::<CartesianScheme>(&y, a, 0.0, l, t, &bc_a, &bc_b, eps, 1000);

    dump_result(&y);
}

//TEST polar dU/dt = 4 * (d2U/dr2 + 1/r dU/dr) ; 0 <= r < 8, 0 < t < T, U(r,0) = 64 - r^2, u(8,t) = 0;
fn test3(nodes: usize, eps: f64) {
    let mut y = vec![0.0; nodes];
    for i in 0..y.len() {
        let x = 8.0 / (y.len() - 1) as f64 * i as f64;
        y[i] = 64.0 - x * x;
    }

    let bc_a = BoundaryCondition2rd { m_du: 0.0 };
    let bc_b = BoundaryCondition1rd { m_u: 0.0 };

    y = explicit_difference_scheme::<PolarScheme>(&y, 4.0, 0.0, 8.0, 5.0, &bc_a, &bc_b, eps, 1000);

    dump_result(&y);
}
