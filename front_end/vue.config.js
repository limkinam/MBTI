module.exports = {
  pluginOptions: {
    s3Deploy: {
      registry: undefined,
      awsProfile: 'default',
      overrideEndpoint: true,
      endpoint: 'limjun92',
      region: 'j',
      bucket: 'j',
      createBucket: true,
      staticHosting: true,
      staticIndexPage: 'index.py',
      staticErrorPage: 'index.j',
      assetPath: 'fj',
      assetMatch: 'fjf',
      deployPath: 'fjf',
      acl: 'public-read-write',
      pwa: false,
      enableCloudfront: false,
      pluginVersion: '4.0.0-rc3',
      uploadConcurrency: 5
    }
  }
}
