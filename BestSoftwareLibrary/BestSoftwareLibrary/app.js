/**
 * Copyright (c) 2022, Sebastien Jodogne, ICTEAM UCLouvain, Belgium
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 **/


var TURTLE_SIZE = 5;
var MARGIN = 4 * TURTLE_SIZE;
var MOVE_STEP = TURTLE_SIZE;
var ANGLE_STEP = 10 / 180 * Math.PI;  // 10 degrees, in radians

var points = [ ];
var turtleX = 0;
var turtleY = 0;
var turtleAngle = 0;  // In radians

function Draw(ctx) {
  if (points.length > 0) {
    ctx.lineWidth *= 3;
    ctx.strokeStyle = 'red';
    ctx.moveTo(points[0][0], points[0][1]);

    for (var i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    
    ctx.lineTo(turtleX, turtleY);
    ctx.stroke();
  }

  var dx = Math.cos(turtleAngle) * TURTLE_SIZE;
  var dy = Math.sin(turtleAngle) * TURTLE_SIZE;

  ctx.beginPath();
  ctx.moveTo(turtleX - dx, turtleY - dy);
  ctx.lineTo(turtleX + dx, turtleY + dy);
  ctx.lineTo(turtleX + 2 * dy, turtleY - 2 * dx);
  ctx.lineTo(turtleX - dx, turtleY - dy);
  ctx.fillStyle = 'gray';
  ctx.fill();

  ctx.beginPath();
  ctx.arc(turtleX, turtleY, TURTLE_SIZE / 5.0, 0, 2.0 * Math.PI);
  ctx.fillStyle = 'black';
  ctx.fill();
}

function GetExtent() {
  var x1 = turtleX;
  var x2 = turtleX;
  var y1 = turtleY;
  var y2 = turtleY;

  for (var i = 0; i < points.length; i++) {
    x1 = Math.min(x1, points[i][0]);
    x2 = Math.max(x2, points[i][0]);
    y1 = Math.min(y1, points[i][1]);
    y2 = Math.max(y2, points[i][1]);
  }
  
  return BestRendering.CreateExtent(x1 - MARGIN, y1 - MARGIN, x2 + MARGIN, y2 + MARGIN);
}

BestRendering.InitializeContainer('turtle', Draw, GetExtent);

document.addEventListener('keydown', function(event) {
  if (event.key == 'ArrowLeft') {
    turtleAngle -= ANGLE_STEP;
  } else if (event.key == 'ArrowRight') {
    turtleAngle += ANGLE_STEP;
  } else if (event.key == 'ArrowUp') {
    points.push([ turtleX, turtleY ]);
    turtleX += MOVE_STEP * Math.sin(turtleAngle);
    turtleY -= MOVE_STEP * Math.cos(turtleAngle);
  } else if (event.key == 'f') {
    BestRendering.FitContainer('turtle');
  } else {
    return;  // Nothing happened
  }

  BestRendering.DrawContainer('turtle');
  event.preventDefault();  // Tag the event as processed
});
