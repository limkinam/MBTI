<template>
  <div id="app">
    <nav>
    <ul class="nav justify-content-center row">
      <li class="navbar-brand">
      <a class="navbar-brand " href="/" style="color:black">
        <img src="./img/mbti3.png" width="35" height="35" class="d-inline-block align-top" >
      </a>
      </li>
      <li class="navbar-brand">
        <a class="nav-link active" href="/" style="color:black">HOME</a>
      </li>
      <li class="navbar-brand">
        <a class="nav-link" href v-on:click="mbti" style="color:black">성격유형</a>
      </li>
      <!-- <li class="navbar-brand">
       <a class="nav-link" href v-on:click="home" id="movetopbt" style="color:black">자기소개서 쓰기</a>
      </li> -->
    </ul>
    </nav>
    <!-- <br><br> -->
    <div class = "start">
    <router-view/>
    </div>
    <footer>
        <p>© 2020 자소서 MBTI : 임기남 최효진 김나라 임준형 강민준 </p>
    </footer>
  </div>
</template>

<script>
export default {
  name: 'App',
  data () {
    return {
      usernameOrEmail: '',
      password: ''
    }
  },
  methods: {
    mbti: function () {
      this.$router.push('/mbti')
    },
    home: function () {
      this.$router.push('/')
    }
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
}
.start{
  background-image: url("./img/back_ground.png");
  background-repeat : no-repeat;
  background-size : cover;
  margin-top:60px
}
.nav{
  text-align: center;
  padding: 10.5px;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background-color: rgba( 201, 235, 243, 1 );
  z-index: 99
}
footer{
  left:0px;
  bottom:0px;
  height:60px;
  width:100%;
  background:grey;
  color: white;
  }
footer p {
    padding: 10.5px;
    margin: 0px;
    line-height: 100%;
}
</style>
