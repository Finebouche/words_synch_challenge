import express from 'express';
export default function() {
  var app = express();
  app.set('port', 3000);
  return app;
};