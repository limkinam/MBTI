import Vue from 'vue'
import Router from 'vue-router'
// import HelloWorld from '@/components/HelloWorld'
import main1 from '@/components/main'
import mbti from '@/components/mbti.vue'
import detail from '@/components/detail'
import HelloWorld from '@/components/HelloWorld'

Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/',
      name: 'main1',
      component: main1
    },
    {
      path: '/mbti',
      name: 'mbti',
      component: mbti
    },
    {
      path: '/detail',
      name: 'detail',
      component: detail
    },
    {
      path: '/hello',
      name: 'hello',
      component: HelloWorld
    }
  ]
})
