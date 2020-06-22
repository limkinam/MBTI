<template>
<div class="container">
  <div class="row">
    <div class="col-6">
    <div style = "width:100%">
        <div style = "width:100%;">
        <pie-chart :data="chartData1" :options="chartOptions"></pie-chart>
        <div>E: {{E}}</div>
        <div>I: {{I}}</div>
        </div>
        <div style = "width:100%; margin-top:50px">
        <pie-chart :data="chartData2" :options="chartOptions"></pie-chart>
        <div>S: {{S}}</div>
        <div>N: {{N}}</div>
        </div>
    </div>
    </div>
    <div class="col-6">
    <div style = "width:100%">
        <div style = "width:100%;">
        <pie-chart :data="chartData3" :options="chartOptions"></pie-chart>
        <div>T: {{T}}</div>
        <div>F: {{F}}</div>
        </div>
        <div style = "width:100%; margin-top:50px">
        <pie-chart :data="chartData4" :options="chartOptions"></pie-chart>
        <div>J: {{J}}</div>
        <div>P: {{P}}</div>
        </div>
    </div>
    </div>
  </div>
  <div class="row" style="margin-top: 80px">
    <div class="col-12">
    <h3>유형을 결정짓는 결정적인 단어들</h3>
    <!-- <h3>E vs I</h3>
    <div style="overflow:hidden;word-wrap:break-word">{{EI}}</div>
    <h3>J vs P</h3>
    <div style="overflow:hidden;word-wrap:break-word">{{JP}}</div>
    <h3>S vs N</h3>
    <div style="overflow:hidden;word-wrap:break-word">{{SN}}</div>
    <h3>T vs F</h3>
    <div style="overflow:hidden;word-wrap:break-word">{{TF}}</div> -->
    <div class="pricing-table pricing-three-column row">
        <div class="plan col-md-6 col-12">
          <div class="plan-name-bronze">
            <h2>E vs I</h2>
          </div>
          <ul>
            <li class="plan-feature"><div v-html="EI"></div></li>
          </ul>
        </div>
        <div style="z-index:55;" class="plan col-md-6 col-12">
          <div class="plan-name-silver">
            <h2>J vs P</h2>
          </div>
          <ul>
            <li class="plan-feature"><div v-html="JP"></div></li>
          </ul>
        </div>
        <div class="plan col-md-6 col-12">
          <div class="plan-name-gold">
            <h2>S vs N</h2>
          </div>
          <ul>
            <li class="plan-feature"><div v-html="SN"></div></li>
          </ul>
        </div>
        <div class="plan col-md-6 col-12">
          <div class="plan-name-blue">
            <h2>T vs F</h2>
          </div>
          <ul>
            <li class="plan-feature"><div v-html="TF"></div></li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>
</template>
<script>
import PieChart from '../PieChart.js'
export default {
  name: 'App',
  components: {
    PieChart
  },
  data () {
    return {
      msg: '',
      E: localStorage.getItem('E'),
      I: localStorage.getItem('I'),
      F: localStorage.getItem('F'),
      T: localStorage.getItem('T'),
      S: localStorage.getItem('S'),
      N: localStorage.getItem('N'),
      J: localStorage.getItem('J'),
      P: localStorage.getItem('P'),
      EI: localStorage.getItem('EI').split(',').join('<br />'),
      JP: localStorage.getItem('JP').split(',').join('<br />'),
      SN: localStorage.getItem('SN').split(',').join('<br />'),
      TF: localStorage.getItem('TF').split(',').join('<br />'),
      chartOptions: {
        hoverBorderWidth: 20
      },
      chartData1: {
        hoverBackgroundColor: 'red',
        hoverBorderWidth: 10,
        labels: ['I(내향성)', 'E(외향성)'],
        datasets: [
          {
            label: 'Data One',
            backgroundColor: ['#F7786B', '#F7CAC9'],
            data: [localStorage.getItem('I'), localStorage.getItem('E')]
          }
        ]
      },
      chartData2: {
        hoverBackgroundColor: 'red',
        hoverBorderWidth: 10,
        labels: ['S(감각형)', 'N(직관형)'],
        datasets: [
          {
            label: 'Data One',
            backgroundColor: ['#91A8D0', '#034F84'],
            data: [localStorage.getItem('S'), localStorage.getItem('N')]
          }
        ]
      },
      chartData3: {
        hoverBackgroundColor: 'red',
        hoverBorderWidth: 10,
        labels: ['T(사고형)', 'F(감정형)'],
        datasets: [
          {
            label: 'Data One',
            backgroundColor: ['#92CFBF', '#C7E2CF'],
            data: [localStorage.getItem('T'), localStorage.getItem('F')]
          }
        ]
      },
      chartData4: {
        hoverBackgroundColor: 'red',
        hoverBorderWidth: 10,
        labels: ['J(판단형)', 'P(인식형)'],
        datasets: [
          {
            label: 'Data One',
            backgroundColor: ['#F3E7DB', '#D9B191'],
            data: [localStorage.getItem('J'), localStorage.getItem('P')]
          }
        ]
      }
    }
  }
  // methods: {
  //   makechart: function () {
  //     console.log(5)
  //   }
  // },
}
</script>
<style scoped>
.container{
  margin-top: 70px;
}
#chart {
    font-family: "Avenir", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
    margin-top: 60px;
}

body{padding-top:20px}

.pricing-table .plan {
  margin-left:0px;
  border-radius: 5px;
  text-align: center;
  background-color: #ffffff;
  -moz-box-shadow: 0 0 6px 2px #b0b2ab;
  -webkit-box-shadow: 0 0 6px 2px #b0b2ab;
  box-shadow: 0 0 6px 2px #b0b2ab;
}

.plan:hover {
  background-color: #fff;
  -moz-box-shadow: 0 0 12px 3px #b0b2ab;
  -webkit-box-shadow: 0 0 12px 3px #b0b2ab;
  box-shadow: 0 0 12px 3px #b0b2ab;
}

.plan {
  padding: 20px;
  margin-left:0px;
  color: black;
  background-color: #5e5f59;
  -moz-border-radius: 5px 5px 0 0;
  -webkit-border-radius: 5px 5px 0 0;
  border-radius: 5px 5px 0 0;
}

.plan-name-bronze {
  padding: 20px;
  color: #fff;
  background-color: #DD6693;
  -moz-border-radius: 5px 5px 0 0;
  -webkit-border-radius: 5px 5px 0 0;
  border-radius: 5px 5px 0 0;
 }

.plan-name-silver {
  padding: 20px;
  color: #fff;
  background-color: #F5CBDE;
  -moz-border-radius: 5px 5px 0 0;
  -webkit-border-radius: 5px 5px 0 0;
  border-radius: 5px 5px 0 0;
 }

.plan-name-gold {
  padding: 20px;
  color: #fff;
  background-color: #9783A6;
  -moz-border-radius: 5px 5px 0 0;
  -webkit-border-radius: 5px 5px 0 0;
  border-radius: 5px 5px 0 0;
}

.plan-name-blue {
  padding: 20px;
  color: #fff;
  background-color: #6C557D;
  -moz-border-radius: 5px 5px 0 0;
  -webkit-border-radius: 5px 5px 0 0;
  border-radius: 5px 5px 0 0;
}

.pricing-table-bronze  {
  padding: 20px;
  color: #fff;
  background-color: #f89406;
  -moz-border-radius: 5px 5px 0 0;
  -webkit-border-radius: 5px 5px 0 0;
  border-radius: 5px 5px 0 0;
}

.pricing-table .plan .plan-name span {
  font-size: 20px;
}

.pricing-table .plan ul {
  list-style: none;
  margin: 0;
  -moz-border-radius: 0 0 5px 5px;
  -webkit-border-radius: 0 0 5px 5px;
  border-radius: 0 0 5px 5px;
}

.pricing-table .plan ul li.plan-feature {
  padding: 15px 10px;
  border-top: 1px solid #c5c8c0;
  margin-right: 35px;
}

.pricing-three-column {
  margin: 0 auto;
  width: 80%;
}

.pricing-variable-height .plan {
  float: none;
  margin-left: 2%;
  vertical-align: bottom;
  display: inline-block;
  zoom:1;
  *display:inline;
}

.plan-mouseover .plan-name {
  background-color: #4e9a06 !important;
}

.btn-plan-select {
  padding: 8px 25px;
  font-size: 18px;
}
</style>
