<div class="column" align="middle">
  <p align="center">
  </p>
  </a>
  <a href="https://github.com/matrixorigin/matrixone/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-red.svg" alt="license"/>
  </a>
  <a href="https://golang.org/">
    <img src="https://img.shields.io/badge/Language-Go-blue.svg" alt="language"/>
  </a>
  <img src="https://img.shields.io/badge/platform-MacOS-white.svg" alt="macos"/>
  <img src="https://img.shields.io/badge/platform-Linux-9cf.svg" alt="linux"/>
  <a href="https://www.codefactor.io/repository/github/matrixorigin/matrixone">
    <img src="https://www.codefactor.io/repository/github/matrixorigin/matrixone/badge?s=7280f4312fca2f2e6938fb8de5b726c5252541f0" alt="codefactor"/>
  </a>
  <a href="https://docs.matrixorigin.cn/en/0.7.0/MatrixOne/Release-Notes/v0.7.0/">
   <img src="https://img.shields.io/badge/Release-v0.7.0-green.svg" alt="release"/>
  </a>

<h5 align="center">If you are interested in This project, please kindly give Me a triple `Star`, `Fork` and `Watch`, Thanks!</h5>

Contents
========

* [What is MatrixOne](#what-is-xxxx)
* [KeyFeatures](#key-features)
* [User Values](#user-values)
* [Architecture](#architecture)
* [Quick start](#quick-start)
* [Contributing](#contributing)
* [License](#license)

## `<a id="what-is-xxxx">`What is XXXX?`</a>`


## 🎯 `<a id="key-features">`Key Features `</a>`

### 💥 **Hyper-converged Engine**

<details>
  <summary><b><font size=4>Monolithic Engine</b></font></summary>
          A monolithic database engine is designed to support hybrid workloads: transactional, analytical, streaming, time-series, machine learning, etc.
</details>

<details>
  <summary><b><font size=4>Built-in Streaming Engine</b></font></summary>
               With the built-in streaming engine, MatrixOne supports in-database streaming processing by groundbreaking incremental materialized view maintenance.
</details>

### ☁️ **Cloud & Edge Native**

<details>
  <summary><b><font size=4>Real Infrastructure Agnostic</b></font></summary>
               MatrixOne supports seemless workload migration and bursting among different locations and infrastructures.
</details>

<details>
  <summary><b><font size=4>Multi-site Active/Active</b></font></summary>
                    MatrixOne provides industry-leading latency control with optimized consistency protocol.
</details>

### 🚀 **Extreme Performance**

<details>
  <summary><b><font size=4>High Performance</b></font></summary>
     Accelerated queries supported by patented vectorized execution as well as optimal computation push down strategies through factorization techniques.
</details>

<details>
  <summary><b><font size=4>Strong Consistency</b></font></summary>
     MatrixOne introduces a global, high-performance distributed transaction protocol across storage engines.
</details>

<details>
  <summary><b><font size=4>High Scalability</b></font></summary>
     Seamless and non-disruptive scaling by disaggregated storage and compute.   
</details>

## 💎 **`<a id="user-values">`User Values `</a>`**

<details>
  <summary><b><font size=4>Simplify Database Management and Maintenance</b></font></summary>
     To solve the problem of high and unpredictable cost of database selection process, management and maintenance due to database overabundance, MatrixOne all-in-one architecture will significantly simplify database management and maintenance, single database can serve multiple data applications.
</details>
<details>
  <summary><b><font size=4>Reduce Data Fragmentation and Inconsistency</b></font></summary>
     Data flow and copy between different databases makes data sync and consistency increasingly difficult. The unified incrementally materialized view of MatrixOne makes the downstream can support real-time upstream update, achieve the end-to-end data processing without redundant ETL process.
</details>
<details>
  <summary><b><font size=4>Decoupling Data Architecture From Infrastructure</b></font></summary>
     Currently the architecture design across different infrastructures is complicated, causes new data silos between cloud and edge, cloud and on-premise. MatrixOne is designed with unified architecture to support simplified data management and operations across different type of infrastructures.
</details>
<details>
  <summary><b><font size=4>Extremely Fast Complex Query Performance</b></font></summary>  
     Poor business agility as a result of slow complex queries and redundant intermediate tables in current data warehousing solutions. MatrixOne  supports blazing fast experience even for star and snowflake schema queries, improving business agility by real-time analytics.
</details>
<details>
  <summary><b><font size=4>A Solid OLTP-like OLAP Experience</b></font></summary>   
     Current data warehousing solutions have the following problems such as high latency and absence of immediate visibility for data updates. MatrixOne brings OLTP (Online Transactional Processing) level consistency and high availability to CRUD operations in OLAP (Online Analytical Processing).
</details>
<details>
  <summary><b><font size=4>Seamless and Non-disruptive Scalability</b></font></summary>   
     It is difficult to balance performance and scalability to achieve optimum price-performance ratio in current data warehousing solutions. MatrixOne's disaggregated storage and compute architecture makes it fully automated and efficient scale in/out and up/down without disrupting applications.
</details>

<br>

## 🔎 `<a id="architecture">`Architecture `</a>`


## ⚡️ `<a id="quick-start">`Quick start `</a>`

### ⚙️ Install MatrixOne

#### **Building from source**

**Step 1. Install Go (version 1.20 is required)**

**Step 2. Get the MatrixOne code to build MatrixOne**

**Step 3. Launch MatrixOne server**

### 🌟 

## 🙌 `<a id="contributing">`Contributing `</a>`

Contributions to MatrixOne are welcome from everyone.
 See [Contribution Guide](https://docs.matrixorigin.cn/en/0.8.0/MatrixOne/Contribution-Guide/make-your-first-contribution/) for details on submitting patches and the contribution workflow.

### 👏 All contributors

<table>
<tr>
    <td align="center">
        <a href="https://github.com/feijianghan">
            <img src="https://avatars.githubusercontent.com/u/88610657?s=96&v=4" width="30;" alt="nnsgmsone"/>
            <br />
            <sub><b>Feijiang Han</b></sub>
        </a>
    </td>
</tr>
</table>
<!-- readme: contributors -end -->

## `<a id="license">`License `</a>`

MatrixOne is licensed under the [Apache License, Version 2.0](LICENSE).
