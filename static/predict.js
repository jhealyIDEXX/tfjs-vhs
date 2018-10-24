var cachedData = [];
const IMAGE_SIZE = 224;
const COORD_SCALE = 5.0;
let vhsModel = null;
let vhsPoints = null;


async function loadTfjs() {
	vhsModel = await tf.loadModel('models/vhs/model.json')
}

$("#image-selector").change(function() {
	let reader = new FileReader();

	reader.onload = function() {
		let dataURL = reader.result;
		$("#image-display").attr("src", dataURL);
	}

	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});


function storePoints(points) {
	for (let i=0;i<points.length;i++) {
		points[i] = points[i] / COORD_SCALE;
	}

	vhsPoints = points;
	drawCanvas();
}

async function predict(imgElement) {

	const logits = tf.tidy(() => {
		const img = tf.fromPixels(imgElement, 1).toFloat()

		const offset = (tf.scalar(255));
		
		const normalized = tf.image.resizeBilinear(img.div(offset), [IMAGE_SIZE, IMAGE_SIZE]);

		const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);

		return vhsModel.predict(batched);
	});

	const values = await logits.data();
	storePoints(values)
}

function predict_image(catElement) {
	let image = new Image();
	image.crossOrigin = 'anonymous';
	image.src = catElement.src;
	image.onload = function() {
		predict(this);
	};
}

function predict_click() {
	let image = $("#image-display")[0];
	console.log('predict click')
	predict_image(image)
}


function drawLine(ctx, x1, y1, x2, y2) {
	ctx.beginPath();
	ctx.moveTo(x1, y1);
	ctx.lineTo(x2, y2);
	ctx.stroke();
}

function distance(x1, y1, x2, y2) {
	let x = x2 - x1;
	let y = y2 - y1;
	return Math.sqrt(x*x + y*y);
}

function drawVhs(ctx, x, y) {
	let minorAxis = distance(vhsPoints[4], vhsPoints[5], vhsPoints[6], vhsPoints[7]);
	let majorAxis = distance(vhsPoints[0], vhsPoints[1], vhsPoints[2], vhsPoints[3]);
	let vertebraeLine = distance(vhsPoints[8], vhsPoints[9], vhsPoints[16], vhsPoints[17]);

	let vhs = (minorAxis+majorAxis)/(vertebraeLine*0.25);
	let text = 'Pred VHS: '+vhs.toFixed(2);
	ctx.fillStyle = 'rgb(250,250,0)';
	ctx.fillText(text, x, y);
}

function drawCanvas() {
	if (!vhsPoints) {
		return;
	}

	var canvas = $("#image-canvas")[0];
	var image = $("#image-display")[0];
	
	let ctx = canvas.getContext('2d');
	let w = image.width;
	let h = image.height;
	canvas.width = w;
	canvas.height = h;

	ctx.drawImage(image, 0, 0);

	ctx.strokeStyle = 'rgb(200,150,0)'

	drawLine(ctx, vhsPoints[0]*w, vhsPoints[1]*h, vhsPoints[2]*w, vhsPoints[3]*h);
	drawLine(ctx, vhsPoints[4]*w, vhsPoints[5]*h, vhsPoints[6]*w, vhsPoints[7]*h);
	drawLine(ctx, vhsPoints[8]*w, vhsPoints[9]*h, vhsPoints[16]*w, vhsPoints[17]*h);

	drawVhs(ctx, vhsPoints[16]*w+20, vhsPoints[17]*h);

	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(vhsPoints[0]*w, vhsPoints[1]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(vhsPoints[2]*w, vhsPoints[3]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(vhsPoints[4]*w, vhsPoints[5]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(vhsPoints[6]*w, vhsPoints[7]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(vhsPoints[8]*w, vhsPoints[9]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(vhsPoints[10]*w, vhsPoints[11]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(vhsPoints[12]*w, vhsPoints[13]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(vhsPoints[14]*w, vhsPoints[15]*h, 2, 2);
	ctx.fillStyle = 'rgb(200,0,0)';
	ctx.fillRect(vhsPoints[16]*w, vhsPoints[17]*h, 2, 2);
	console.log(vhsPoints);

}

loadTfjs();