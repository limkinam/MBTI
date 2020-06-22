import axios from 'axios'

export default axios.create({
  baseURL: 'http://15.165.18.70:5000/',
  headers: {
    'Content-type': 'application/json'
  }
})
