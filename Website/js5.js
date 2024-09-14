if(window.innerHeight > window.innerWidth){
	document.write('<link rel="stylesheet" href="stylesmob.css">')
} else {
	document.write('<link rel="stylesheet" href="styles.css">')
}
document.write('<link rel="apple-touch-icon" sizes="180x180" href="https://captchacoin.net/apple-touch-icon.png"><link rel="icon" type="image/png" sizes="32x32" href="https://captchacoin.net/favicon-32x32.png"><link rel="icon" type="image/png" sizes="16x16" href="https://captchacoin.net/favicon-16x16.png"><link rel="icon" href="https://captchacoin.net/favicon.ico" type="image/png"><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=M+PLUS+2:wght@400;700&display=swap" rel="stylesheet">');
function setCookie(name,value,days) {
    var expires = "";
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days*24*60*60*1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "")  + expires + "; path=/";
}

function readCookie(name) {
	var nameEQ = name + "=";
	var ca = document.cookie.split(';');
	for(var i=0;i < ca.length;i++) {
		var c = ca[i];
		while (c.charAt(0)==' ') c = c.substring(1,c.length);
		if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
	}
	return null;
}

const zeroPad = (num, places) => String(num).padStart(places, '0');
var username = readCookie('username');
var caps = readCookie('caps');
var logs = readCookie('logs');
var lastmine = readCookie('lastmine');
var mined = readCookie('mined');
var lmined = readCookie('lmined');
var rmined = readCookie('rmined');
var dif = readCookie('dif');
var difm = readCookie('difm');
var difspwb = readCookie('difspwb');
var difspwb = readCookie('diftipb');
var ref = readCookie('ref');
var refm = readCookie('refm');
var refspwb = readCookie('refspwb');
var refspwb = readCookie('reftipb');
var pos = readCookie('pos');
var sol = readCookie('sol');
var fwd = readCookie('fwd');
var trans = readCookie('trans');
var prkey = readCookie('prkey');
var quickusername = readCookie('quickusername');
var quickusernamesecret = readCookie('quickusernamesecret');

logoutbar = '<center><div class="wrap"><div class="one"><a href="https://captchacoin.net/earn/my-wallet.php"><img src="https://captchacoin.net/im/wallet.png" class="icon"><h2 style="color:#00cbbd">'+username+'&rsquo;s Wallet</h2><h3 style="color:#00cbbd">You have ' + (Math.round(caps/1000000000)).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",") + ' caps</h3></a></div><div class="two"><a href="https://captchacoin.net/earn/logout.html"><img src="https://captchacoin.net/im/logout.png"class="icon"><h2 style="color:#c2c2c2">Logout</h2></a></div></div></center>';
