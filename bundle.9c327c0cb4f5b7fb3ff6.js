(()=>{"use strict";function t(){}function n(n){return null==n?t:function(){return this.querySelector(n)}}function e(){return[]}function r(t){return function(n){return n.matches(t)}}var i=Array.prototype.find;function o(){return this.firstElementChild}var u=Array.prototype.filter;function s(){return Array.from(this.children)}function a(t){return new Array(t.length)}function h(t,n){this.ownerDocument=t.ownerDocument,this.namespaceURI=t.namespaceURI,this._next=null,this._parent=t,this.__data__=n}function l(t,n,e,r,i,o){for(var u,s=0,a=n.length,l=o.length;s<l;++s)(u=n[s])?(u.__data__=o[s],r[s]=u):e[s]=new h(t,o[s]);for(;s<a;++s)(u=n[s])&&(i[s]=u)}function c(t,n,e,r,i,o,u){var s,a,l,c=new Map,f=n.length,p=o.length,_=new Array(f);for(s=0;s<f;++s)(a=n[s])&&(_[s]=l=u.call(a,a.__data__,s,n)+"",c.has(l)?i[s]=a:c.set(l,a));for(s=0;s<p;++s)l=u.call(t,o[s],s,o)+"",(a=c.get(l))?(r[s]=a,a.__data__=o[s],c.delete(l)):e[s]=new h(t,o[s]);for(s=0;s<f;++s)(a=n[s])&&c.get(_[s])===a&&(i[s]=a)}function f(t){return t.__data__}function p(t){return"object"==typeof t&&"length"in t?t:Array.from(t)}function _(t,n){return t<n?-1:t>n?1:t>=n?0:NaN}h.prototype={constructor:h,appendChild:function(t){return this._parent.insertBefore(t,this._next)},insertBefore:function(t,n){return this._parent.insertBefore(t,n)},querySelector:function(t){return this._parent.querySelector(t)},querySelectorAll:function(t){return this._parent.querySelectorAll(t)}};var d="http://www.w3.org/1999/xhtml";const y={svg:"http://www.w3.org/2000/svg",xhtml:d,xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace",xmlns:"http://www.w3.org/2000/xmlns/"};function v(t){var n=t+="",e=n.indexOf(":");return e>=0&&"xmlns"!==(n=t.slice(0,e))&&(t=t.slice(e+1)),y.hasOwnProperty(n)?{space:y[n],local:t}:t}function g(t){return function(){this.removeAttribute(t)}}function m(t){return function(){this.removeAttributeNS(t.space,t.local)}}function w(t,n){return function(){this.setAttribute(t,n)}}function x(t,n){return function(){this.setAttributeNS(t.space,t.local,n)}}function A(t,n){return function(){var e=n.apply(this,arguments);null==e?this.removeAttribute(t):this.setAttribute(t,e)}}function $(t,n){return function(){var e=n.apply(this,arguments);null==e?this.removeAttributeNS(t.space,t.local):this.setAttributeNS(t.space,t.local,e)}}function E(t){return t.ownerDocument&&t.ownerDocument.defaultView||t.document&&t||t.defaultView}function b(t){return function(){this.style.removeProperty(t)}}function S(t,n,e){return function(){this.style.setProperty(t,n,e)}}function M(t,n,e){return function(){var r=n.apply(this,arguments);null==r?this.style.removeProperty(t):this.style.setProperty(t,r,e)}}function N(t){return function(){delete this[t]}}function z(t,n){return function(){this[t]=n}}function B(t,n){return function(){var e=n.apply(this,arguments);null==e?delete this[t]:this[t]=e}}function C(t){return t.trim().split(/^|\s+/)}function L(t){return t.classList||new T(t)}function T(t){this._node=t,this._names=C(t.getAttribute("class")||"")}function P(t,n){for(var e=L(t),r=-1,i=n.length;++r<i;)e.add(n[r])}function D(t,n){for(var e=L(t),r=-1,i=n.length;++r<i;)e.remove(n[r])}function O(t){return function(){P(this,t)}}function q(t){return function(){D(this,t)}}function I(t,n){return function(){(n.apply(this,arguments)?P:D)(this,t)}}function k(){this.textContent=""}function R(t){return function(){this.textContent=t}}function j(t){return function(){var n=t.apply(this,arguments);this.textContent=null==n?"":n}}function H(){this.innerHTML=""}function J(t){return function(){this.innerHTML=t}}function U(t){return function(){var n=t.apply(this,arguments);this.innerHTML=null==n?"":n}}function V(){this.nextSibling&&this.parentNode.appendChild(this)}function F(){this.previousSibling&&this.parentNode.insertBefore(this,this.parentNode.firstChild)}function Z(t){return function(){var n=this.ownerDocument,e=this.namespaceURI;return e===d&&n.documentElement.namespaceURI===d?n.createElement(t):n.createElementNS(e,t)}}function Q(t){return function(){return this.ownerDocument.createElementNS(t.space,t.local)}}function X(t){var n=v(t);return(n.local?Q:Z)(n)}function G(){return null}function K(){var t=this.parentNode;t&&t.removeChild(this)}function W(){var t=this.cloneNode(!1),n=this.parentNode;return n?n.insertBefore(t,this.nextSibling):t}function Y(){var t=this.cloneNode(!0),n=this.parentNode;return n?n.insertBefore(t,this.nextSibling):t}function tt(t){return function(){var n=this.__on;if(n){for(var e,r=0,i=-1,o=n.length;r<o;++r)e=n[r],t.type&&e.type!==t.type||e.name!==t.name?n[++i]=e:this.removeEventListener(e.type,e.listener,e.options);++i?n.length=i:delete this.__on}}}function nt(t,n,e){return function(){var r,i=this.__on,o=function(t){return function(n){t.call(this,n,this.__data__)}}(n);if(i)for(var u=0,s=i.length;u<s;++u)if((r=i[u]).type===t.type&&r.name===t.name)return this.removeEventListener(r.type,r.listener,r.options),this.addEventListener(r.type,r.listener=o,r.options=e),void(r.value=n);this.addEventListener(t.type,o,e),r={type:t.type,name:t.name,value:n,listener:o,options:e},i?i.push(r):this.__on=[r]}}function et(t,n,e){var r=E(t),i=r.CustomEvent;"function"==typeof i?i=new i(n,e):(i=r.document.createEvent("Event"),e?(i.initEvent(n,e.bubbles,e.cancelable),i.detail=e.detail):i.initEvent(n,!1,!1)),t.dispatchEvent(i)}function rt(t,n){return function(){return et(this,t,n)}}function it(t,n){return function(){return et(this,t,n.apply(this,arguments))}}T.prototype={add:function(t){this._names.indexOf(t)<0&&(this._names.push(t),this._node.setAttribute("class",this._names.join(" ")))},remove:function(t){var n=this._names.indexOf(t);n>=0&&(this._names.splice(n,1),this._node.setAttribute("class",this._names.join(" ")))},contains:function(t){return this._names.indexOf(t)>=0}};var ot=[null];function ut(t,n){this._groups=t,this._parents=n}function st(t){var n=0,e=t.children,r=e&&e.length;if(r)for(;--r>=0;)n+=e[r].value;else n=1;t.value=n}function at(t,n){t instanceof Map?(t=[void 0,t],void 0===n&&(n=lt)):void 0===n&&(n=ht);for(var e,r,i,o,u,s=new pt(t),a=[s];e=a.pop();)if((i=n(e.data))&&(u=(i=Array.from(i)).length))for(e.children=i,o=u-1;o>=0;--o)a.push(r=i[o]=new pt(i[o])),r.parent=e,r.depth=e.depth+1;return s.eachBefore(ft)}function ht(t){return t.children}function lt(t){return Array.isArray(t)?t[1]:null}function ct(t){void 0!==t.data.value&&(t.value=t.data.value),t.data=t.data.data}function ft(t){var n=0;do{t.height=n}while((t=t.parent)&&t.height<++n)}function pt(t){this.data=t,this.depth=this.height=0,this.parent=null}function _t(t,n){return t.parent===n.parent?1:2}function dt(t){var n=t.children;return n?n[0]:t.t}function yt(t){var n=t.children;return n?n[n.length-1]:t.t}function vt(t,n,e){var r=e/(n.i-t.i);n.c-=r,n.s+=e,t.c+=r,n.z+=e,n.m+=e}function gt(t,n,e){return t.a.parent===n.parent?t.a:e}function mt(t,n){this._=t,this.parent=null,this.children=null,this.A=null,this.a=this,this.z=0,this.m=0,this.c=0,this.s=0,this.t=null,this.i=n}ut.prototype=function(){return new ut([[document.documentElement]],ot)}.prototype={constructor:ut,select:function(t){"function"!=typeof t&&(t=n(t));for(var e=this._groups,r=e.length,i=new Array(r),o=0;o<r;++o)for(var u,s,a=e[o],h=a.length,l=i[o]=new Array(h),c=0;c<h;++c)(u=a[c])&&(s=t.call(u,u.__data__,c,a))&&("__data__"in u&&(s.__data__=u.__data__),l[c]=s);return new ut(i,this._parents)},selectAll:function(t){t="function"==typeof t?function(t){return function(){return null==(n=t.apply(this,arguments))?[]:Array.isArray(n)?n:Array.from(n);var n}}(t):function(t){return null==t?e:function(){return this.querySelectorAll(t)}}(t);for(var n=this._groups,r=n.length,i=[],o=[],u=0;u<r;++u)for(var s,a=n[u],h=a.length,l=0;l<h;++l)(s=a[l])&&(i.push(t.call(s,s.__data__,l,a)),o.push(s));return new ut(i,o)},selectChild:function(t){return this.select(null==t?o:function(t){return function(){return i.call(this.children,t)}}("function"==typeof t?t:r(t)))},selectChildren:function(t){return this.selectAll(null==t?s:function(t){return function(){return u.call(this.children,t)}}("function"==typeof t?t:r(t)))},filter:function(t){"function"!=typeof t&&(t=function(t){return function(){return this.matches(t)}}(t));for(var n=this._groups,e=n.length,r=new Array(e),i=0;i<e;++i)for(var o,u=n[i],s=u.length,a=r[i]=[],h=0;h<s;++h)(o=u[h])&&t.call(o,o.__data__,h,u)&&a.push(o);return new ut(r,this._parents)},data:function(t,n){if(!arguments.length)return Array.from(this,f);var e,r=n?c:l,i=this._parents,o=this._groups;"function"!=typeof t&&(e=t,t=function(){return e});for(var u=o.length,s=new Array(u),a=new Array(u),h=new Array(u),_=0;_<u;++_){var d=i[_],y=o[_],v=y.length,g=p(t.call(d,d&&d.__data__,_,i)),m=g.length,w=a[_]=new Array(m),x=s[_]=new Array(m);r(d,y,w,x,h[_]=new Array(v),g,n);for(var A,$,E=0,b=0;E<m;++E)if(A=w[E]){for(E>=b&&(b=E+1);!($=x[b])&&++b<m;);A._next=$||null}}return(s=new ut(s,i))._enter=a,s._exit=h,s},enter:function(){return new ut(this._enter||this._groups.map(a),this._parents)},exit:function(){return new ut(this._exit||this._groups.map(a),this._parents)},join:function(t,n,e){var r=this.enter(),i=this,o=this.exit();return"function"==typeof t?(r=t(r))&&(r=r.selection()):r=r.append(t+""),null!=n&&(i=n(i))&&(i=i.selection()),null==e?o.remove():e(o),r&&i?r.merge(i).order():i},merge:function(t){for(var n=t.selection?t.selection():t,e=this._groups,r=n._groups,i=e.length,o=r.length,u=Math.min(i,o),s=new Array(i),a=0;a<u;++a)for(var h,l=e[a],c=r[a],f=l.length,p=s[a]=new Array(f),_=0;_<f;++_)(h=l[_]||c[_])&&(p[_]=h);for(;a<i;++a)s[a]=e[a];return new ut(s,this._parents)},selection:function(){return this},order:function(){for(var t=this._groups,n=-1,e=t.length;++n<e;)for(var r,i=t[n],o=i.length-1,u=i[o];--o>=0;)(r=i[o])&&(u&&4^r.compareDocumentPosition(u)&&u.parentNode.insertBefore(r,u),u=r);return this},sort:function(t){function n(n,e){return n&&e?t(n.__data__,e.__data__):!n-!e}t||(t=_);for(var e=this._groups,r=e.length,i=new Array(r),o=0;o<r;++o){for(var u,s=e[o],a=s.length,h=i[o]=new Array(a),l=0;l<a;++l)(u=s[l])&&(h[l]=u);h.sort(n)}return new ut(i,this._parents).order()},call:function(){var t=arguments[0];return arguments[0]=this,t.apply(null,arguments),this},nodes:function(){return Array.from(this)},node:function(){for(var t=this._groups,n=0,e=t.length;n<e;++n)for(var r=t[n],i=0,o=r.length;i<o;++i){var u=r[i];if(u)return u}return null},size:function(){let t=0;for(const n of this)++t;return t},empty:function(){return!this.node()},each:function(t){for(var n=this._groups,e=0,r=n.length;e<r;++e)for(var i,o=n[e],u=0,s=o.length;u<s;++u)(i=o[u])&&t.call(i,i.__data__,u,o);return this},attr:function(t,n){var e=v(t);if(arguments.length<2){var r=this.node();return e.local?r.getAttributeNS(e.space,e.local):r.getAttribute(e)}return this.each((null==n?e.local?m:g:"function"==typeof n?e.local?$:A:e.local?x:w)(e,n))},style:function(t,n,e){return arguments.length>1?this.each((null==n?b:"function"==typeof n?M:S)(t,n,null==e?"":e)):function(t,n){return t.style.getPropertyValue(n)||E(t).getComputedStyle(t,null).getPropertyValue(n)}(this.node(),t)},property:function(t,n){return arguments.length>1?this.each((null==n?N:"function"==typeof n?B:z)(t,n)):this.node()[t]},classed:function(t,n){var e=C(t+"");if(arguments.length<2){for(var r=L(this.node()),i=-1,o=e.length;++i<o;)if(!r.contains(e[i]))return!1;return!0}return this.each(("function"==typeof n?I:n?O:q)(e,n))},text:function(t){return arguments.length?this.each(null==t?k:("function"==typeof t?j:R)(t)):this.node().textContent},html:function(t){return arguments.length?this.each(null==t?H:("function"==typeof t?U:J)(t)):this.node().innerHTML},raise:function(){return this.each(V)},lower:function(){return this.each(F)},append:function(t){var n="function"==typeof t?t:X(t);return this.select((function(){return this.appendChild(n.apply(this,arguments))}))},insert:function(t,e){var r="function"==typeof t?t:X(t),i=null==e?G:"function"==typeof e?e:n(e);return this.select((function(){return this.insertBefore(r.apply(this,arguments),i.apply(this,arguments)||null)}))},remove:function(){return this.each(K)},clone:function(t){return this.select(t?Y:W)},datum:function(t){return arguments.length?this.property("__data__",t):this.node().__data__},on:function(t,n,e){var r,i,o=function(t){return t.trim().split(/^|\s+/).map((function(t){var n="",e=t.indexOf(".");return e>=0&&(n=t.slice(e+1),t=t.slice(0,e)),{type:t,name:n}}))}(t+""),u=o.length;if(!(arguments.length<2)){for(s=n?nt:tt,r=0;r<u;++r)this.each(s(o[r],n,e));return this}var s=this.node().__on;if(s)for(var a,h=0,l=s.length;h<l;++h)for(r=0,a=s[h];r<u;++r)if((i=o[r]).type===a.type&&i.name===a.name)return a.value},dispatch:function(t,n){return this.each(("function"==typeof n?it:rt)(t,n))},[Symbol.iterator]:function*(){for(var t=this._groups,n=0,e=t.length;n<e;++n)for(var r,i=t[n],o=0,u=i.length;o<u;++o)(r=i[o])&&(yield r)}},pt.prototype=at.prototype={constructor:pt,count:function(){return this.eachAfter(st)},each:function(t,n){let e=-1;for(const r of this)t.call(n,r,++e,this);return this},eachAfter:function(t,n){for(var e,r,i,o=this,u=[o],s=[],a=-1;o=u.pop();)if(s.push(o),e=o.children)for(r=0,i=e.length;r<i;++r)u.push(e[r]);for(;o=s.pop();)t.call(n,o,++a,this);return this},eachBefore:function(t,n){for(var e,r,i=this,o=[i],u=-1;i=o.pop();)if(t.call(n,i,++u,this),e=i.children)for(r=e.length-1;r>=0;--r)o.push(e[r]);return this},find:function(t,n){let e=-1;for(const r of this)if(t.call(n,r,++e,this))return r},sum:function(t){return this.eachAfter((function(n){for(var e=+t(n.data)||0,r=n.children,i=r&&r.length;--i>=0;)e+=r[i].value;n.value=e}))},sort:function(t){return this.eachBefore((function(n){n.children&&n.children.sort(t)}))},path:function(t){for(var n=this,e=function(t,n){if(t===n)return t;var e=t.ancestors(),r=n.ancestors(),i=null;for(t=e.pop(),n=r.pop();t===n;)i=t,t=e.pop(),n=r.pop();return i}(n,t),r=[n];n!==e;)n=n.parent,r.push(n);for(var i=r.length;t!==e;)r.splice(i,0,t),t=t.parent;return r},ancestors:function(){for(var t=this,n=[t];t=t.parent;)n.push(t);return n},descendants:function(){return Array.from(this)},leaves:function(){var t=[];return this.eachBefore((function(n){n.children||t.push(n)})),t},links:function(){var t=this,n=[];return t.each((function(e){e!==t&&n.push({source:e.parent,target:e})})),n},copy:function(){return at(this).eachBefore(ct)},[Symbol.iterator]:function*(){var t,n,e,r,i=this,o=[i];do{for(t=o.reverse(),o=[];i=t.pop();)if(yield i,n=i.children)for(e=0,r=n.length;e<r;++e)o.push(n[e])}while(o.length)}},mt.prototype=Object.create(pt.prototype);var wt=Array.prototype.slice;function xt(t){return function(){return t}}class At{constructor(t,n){this._context=t,this._x=n}areaStart(){this._line=0}areaEnd(){this._line=NaN}lineStart(){this._point=0}lineEnd(){(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line}point(t,n){switch(t=+t,n=+n,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,n):this._context.moveTo(t,n);break;case 1:this._point=2;default:this._x?this._context.bezierCurveTo(this._x0=(this._x0+t)/2,this._y0,this._x0,n,t,n):this._context.bezierCurveTo(this._x0,this._y0=(this._y0+n)/2,t,this._y0,t,n)}this._x0=t,this._y0=n}}function $t(t){return new At(t,!0)}const Et=Math.PI,bt=2*Et,St=1e-6,Mt=bt-St;function Nt(t){this._+=t[0];for(let n=1,e=t.length;n<e;++n)this._+=arguments[n]+t[n]}class zt{constructor(t){this._x0=this._y0=this._x1=this._y1=null,this._="",this._append=null==t?Nt:function(t){let n=Math.floor(t);if(!(n>=0))throw new Error(`invalid digits: ${t}`);if(n>15)return Nt;const e=10**n;return function(t){this._+=t[0];for(let n=1,r=t.length;n<r;++n)this._+=Math.round(arguments[n]*e)/e+t[n]}}(t)}moveTo(t,n){this._append`M${this._x0=this._x1=+t},${this._y0=this._y1=+n}`}closePath(){null!==this._x1&&(this._x1=this._x0,this._y1=this._y0,this._append`Z`)}lineTo(t,n){this._append`L${this._x1=+t},${this._y1=+n}`}quadraticCurveTo(t,n,e,r){this._append`Q${+t},${+n},${this._x1=+e},${this._y1=+r}`}bezierCurveTo(t,n,e,r,i,o){this._append`C${+t},${+n},${+e},${+r},${this._x1=+i},${this._y1=+o}`}arcTo(t,n,e,r,i){if(t=+t,n=+n,e=+e,r=+r,(i=+i)<0)throw new Error(`negative radius: ${i}`);let o=this._x1,u=this._y1,s=e-t,a=r-n,h=o-t,l=u-n,c=h*h+l*l;if(null===this._x1)this._append`M${this._x1=t},${this._y1=n}`;else if(c>St)if(Math.abs(l*s-a*h)>St&&i){let f=e-o,p=r-u,_=s*s+a*a,d=f*f+p*p,y=Math.sqrt(_),v=Math.sqrt(c),g=i*Math.tan((Et-Math.acos((_+c-d)/(2*y*v)))/2),m=g/v,w=g/y;Math.abs(m-1)>St&&this._append`L${t+m*h},${n+m*l}`,this._append`A${i},${i},0,0,${+(l*f>h*p)},${this._x1=t+w*s},${this._y1=n+w*a}`}else this._append`L${this._x1=t},${this._y1=n}`}arc(t,n,e,r,i,o){if(t=+t,n=+n,o=!!o,(e=+e)<0)throw new Error(`negative radius: ${e}`);let u=e*Math.cos(r),s=e*Math.sin(r),a=t+u,h=n+s,l=1^o,c=o?r-i:i-r;null===this._x1?this._append`M${a},${h}`:(Math.abs(this._x1-a)>St||Math.abs(this._y1-h)>St)&&this._append`L${a},${h}`,e&&(c<0&&(c=c%bt+bt),c>Mt?this._append`A${e},${e},0,1,${l},${t-u},${n-s}A${e},${e},0,1,${l},${this._x1=a},${this._y1=h}`:c>St&&this._append`A${e},${e},0,${+(c>=Et)},${l},${this._x1=t+e*Math.cos(i)},${this._y1=n+e*Math.sin(i)}`)}rect(t,n,e,r){this._append`M${this._x0=this._x1=+t},${this._y0=this._y1=+n}h${e=+e}v${+r}h${-e}Z`}toString(){return this._}}function Bt(t){return t[0]}function Ct(t){return t[1]}function Lt(t){return t.source}function Tt(t){return t.target}const Pt=new class{constructor(){this.data=null}loadFile(t){return new Promise(((n,e)=>{const r=new FileReader;r.onload=t=>{try{this.data=JSON.parse(t.target.result),n(this.data)}catch(t){e(new Error("Error parsing JSON: "+t))}},r.onerror=()=>{e(new Error("Error reading file"))},r.readAsText(t)}))}getData(){return this.data}hasData(){return null!==this.data}clearData(){this.data=null}},Dt=function(t){return new ut([[document.querySelector(t)]],[document.documentElement])}("svg");let Ot=at("");function qt(t,n){(function(){var t=_t,n=1,e=1,r=null;function i(i){var a=function(t){for(var n,e,r,i,o,u=new mt(t,0),s=[u];n=s.pop();)if(r=n._.children)for(n.children=new Array(o=r.length),i=o-1;i>=0;--i)s.push(e=n.children[i]=new mt(r[i],i)),e.parent=n;return(u.parent=new mt(null,0)).children=[u],u}(i);if(a.eachAfter(o),a.parent.m=-a.z,a.eachBefore(u),r)i.eachBefore(s);else{var h=i,l=i,c=i;i.eachBefore((function(t){t.x<h.x&&(h=t),t.x>l.x&&(l=t),t.depth>c.depth&&(c=t)}));var f=h===l?1:t(h,l)/2,p=f-h.x,_=n/(l.x+f+p),d=e/(c.depth||1);i.eachBefore((function(t){t.x=(t.x+p)*_,t.y=t.depth*d}))}return i}function o(n){var e=n.children,r=n.parent.children,i=n.i?r[n.i-1]:null;if(e){!function(t){for(var n,e=0,r=0,i=t.children,o=i.length;--o>=0;)(n=i[o]).z+=e,n.m+=e,e+=n.s+(r+=n.c)}(n);var o=(e[0].z+e[e.length-1].z)/2;i?(n.z=i.z+t(n._,i._),n.m=n.z-o):n.z=o}else i&&(n.z=i.z+t(n._,i._));n.parent.A=function(n,e,r){if(e){for(var i,o=n,u=n,s=e,a=o.parent.children[0],h=o.m,l=u.m,c=s.m,f=a.m;s=yt(s),o=dt(o),s&&o;)a=dt(a),(u=yt(u)).a=n,(i=s.z+c-o.z-h+t(s._,o._))>0&&(vt(gt(s,n,r),n,i),h+=i,l+=i),c+=s.m,h+=o.m,f+=a.m,l+=u.m;s&&!yt(u)&&(u.t=s,u.m+=c-l),o&&!dt(a)&&(a.t=o,a.m+=h-f,r=n)}return r}(n,i,n.parent.A||r[0])}function u(t){t._.x=t.z+t.parent.m,t.m+=t.parent.m}function s(t){t.x*=n,t.y=t.depth*e}return i.separation=function(n){return arguments.length?(t=n,i):t},i.size=function(t){return arguments.length?(r=!1,n=+t[0],e=+t[1],i):r?null:[n,e]},i.nodeSize=function(t){return arguments.length?(r=!0,n=+t[0],e=+t[1],i):r?[n,e]:null},i})().size([600,900])(t);const e=n.append("g").attr("transform","translate(40,40)"),r=function(t){let n=Lt,e=Tt,r=Bt,i=Ct,o=null,u=null,s=function(t){let n=3;return t.digits=function(e){if(!arguments.length)return n;if(null==e)n=null;else{const t=Math.floor(e);if(!(t>=0))throw new RangeError(`invalid digits: ${e}`);n=t}return t},()=>new zt(n)}(a);function a(){let a;const h=wt.call(arguments),l=n.apply(this,h),c=e.apply(this,h);if(null==o&&(u=t(a=s())),u.lineStart(),h[0]=l,u.point(+r.apply(this,h),+i.apply(this,h)),h[0]=c,u.point(+r.apply(this,h),+i.apply(this,h)),u.lineEnd(),a)return u=null,a+""||null}return a.source=function(t){return arguments.length?(n=t,a):n},a.target=function(t){return arguments.length?(e=t,a):e},a.x=function(t){return arguments.length?(r="function"==typeof t?t:xt(+t),a):r},a.y=function(t){return arguments.length?(i="function"==typeof t?t:xt(+t),a):i},a.context=function(n){return arguments.length?(null==n?o=u=null:u=t(o=n),a):o},a}($t).x((t=>t.y)).y((t=>t.x));e.selectAll(".link").data(t.links()).enter().append("path").attr("class","link").attr("d",r);const i=e.selectAll(".node").data(t.descendants()).enter().append("g").attr("class","node").attr("transform",(t=>`translate(${t.y},${t.x})`));i.append("circle").attr("r",5),i.append("text").attr("dy",3).attr("x",(t=>t.children?-10:10)).style("text-anchor",(t=>t.children?"end":"start")).text((t=>t.data.name))}document.getElementById("loadButton").addEventListener("click",(()=>{document.getElementById("fileInput").click()})),document.getElementById("fileInput").addEventListener("change",(t=>{const n=t.target.files[0];n&&Pt.loadFile(n).then((t=>{Ot=at(Pt.getData()),console.log("Loaded JSON:",t),qt(Ot,Dt)})).catch((t=>{console.error("Error parsing JSON:",t)}))})),qt(Ot,Dt)})();